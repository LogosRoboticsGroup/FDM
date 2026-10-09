from __future__ import annotations

import asyncio
import json
import logging
import ssl
import threading
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, urlsplit

from websockets.asyncio.server import ServerConnection, serve
from websockets.http11 import Request, Response

from ..config import VRConfig
from .protocol import VRState, parse_vr_message

logger = logging.getLogger(__name__)


class _StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, websocket_port: int, **kwargs):
        self.websocket_port = websocket_port
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/runtime-config.js":
            body = f"window.STARVLA_VR = {{ websocketPort: {self.websocket_port} }};\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt: str, *args) -> None:
        logger.debug("WebXR HTTP: " + fmt, *args)


class WebXRServer:
    def __init__(self, config: VRConfig, state: VRState | None = None):
        self.config = config
        self.state = state or VRState()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._http: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._ws_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return bool(self._ws_thread and self._ws_thread.is_alive())

    def start(self, timeout_s: float = 5.0) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._ready.clear()
        self._start_http()
        self._ws_thread = threading.Thread(target=lambda: asyncio.run(self._serve_ws()), name="webxr-ws", daemon=True)
        self._ws_thread.start()
        if not self._ready.wait(timeout_s):
            self.stop()
            raise TimeoutError("WebXR server did not start")
        logger.info("Open %s in the VR browser", self._browser_url(self.config.bind_host))
        logger.info(
            "WebXR controller WebSocket listening on %s://%s:%d/vr (do not open this URL directly)",
            "wss" if self.config.tls_cert_path else "ws",
            self.config.bind_host,
            self.config.websocket_port,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
            self._http = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=2.0)
            self._http_thread = None
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=2.0)
            self._ws_thread = None

    def _ssl_context(self) -> ssl.SSLContext | None:
        cert, key = self.config.tls_cert_path, self.config.tls_key_path
        if not cert and not key:
            return None
        if not cert or not key:
            raise ValueError("both vr.tls_cert_path and vr.tls_key_path are required")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        return context

    def _start_http(self) -> None:
        web_root = Path(__file__).with_name("web_ui")
        handler = partial(
            _StaticHandler,
            directory=str(web_root),
            websocket_port=self.config.websocket_port,
        )
        self._http = ThreadingHTTPServer((self.config.bind_host, self.config.https_port), handler)
        context = self._ssl_context()
        if context is not None:
            self._http.socket = context.wrap_socket(self._http.socket, server_side=True)
        self._http_thread = threading.Thread(target=self._http.serve_forever, name="webxr-http", daemon=True)
        self._http_thread.start()

    def _browser_url(self, hostname: str) -> str:
        scheme = "https" if self.config.tls_cert_path else "http"
        formatted_host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
        return f"{scheme}://{formatted_host}:{self.config.https_port}/"

    def _redirect_browser_request(
        self,
        connection: ServerConnection,
        request: Request,
    ) -> Response | None:
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None

        request_host = request.headers.get("Host", "")
        try:
            hostname = urlsplit(f"//{request_host}").hostname
        except ValueError:
            hostname = None
        location = self._browser_url(hostname or self.config.bind_host)
        response = connection.respond(
            HTTPStatus.TEMPORARY_REDIRECT,
            f"This port accepts WebSocket clients. Open {location} in the VR browser.\n",
        )
        response.headers["Location"] = location
        return response

    async def _serve_ws(self) -> None:
        origins = list(self.config.allowed_origins) or None
        async with serve(
            self._handle_connection,
            self.config.bind_host,
            self.config.websocket_port,
            ssl=self._ssl_context(),
            origins=origins,
            process_request=self._redirect_browser_request,
            max_size=64 * 1024,
            max_queue=2,
            compression=None,
        ):
            self._ready.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.1)

    async def _handle_connection(self, connection: ServerConnection) -> None:
        session_id: str | None = None
        last_seq: int | None = None
        try:
            async for message in connection:
                raw = json.loads(message)
                if raw.get("type") == "event":
                    event = str(raw.get("event"))
                    self.state.push_event(event)
                    logger.info("WebXR operator event: %s", event)
                    continue
                frame = parse_vr_message(raw)
                session_id = frame.session_id
                last_seq = frame.seq
                self.state.update(frame)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Ignoring invalid WebXR message: %s", exc)
        finally:
            if session_id is not None:
                self.state.disconnect(session_id, last_seq)
