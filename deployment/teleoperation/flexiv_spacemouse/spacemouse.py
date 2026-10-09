from __future__ import annotations

import errno
import fcntl
import logging
import os
import select
import struct
import threading
import time
from pathlib import Path

from .config import SpaceMouseConfig
from .contracts import SpaceMouseState

logger = logging.getLogger(__name__)

EV_SYN = 0x00
EV_KEY = 0x01
EV_ABS = 0x03
SYN_REPORT = 0
ABS_CODES = tuple(range(6))
BTN_MISC = 0x100
INPUT_EVENT = struct.Struct("llHHi")
ABS_INFO = struct.Struct("iiiiii")
USBDEVFS_IOCTL_ARGUMENT = struct.Struct("iiP")
USBDEVFS_CONNECT = (ord("U") << 8) | 23
USBDEVFS_IOCTL = (
    (3 << 30)
    | (USBDEVFS_IOCTL_ARGUMENT.size << 16)
    | (ord("U") << 8)
    | 18
)


def _ev_iocgabs(axis_code: int) -> int:
    # Linux _IOR('E', 0x40 + axis, struct input_absinfo).
    return (2 << 30) | (ABS_INFO.size << 16) | (ord("E") << 8) | (0x40 + axis_code)


def _matching_event_devices(
    name_contains: str,
    *,
    sys_input_root: Path = Path("/sys/class/input"),
    dev_input_root: Path = Path("/dev/input"),
) -> list[tuple[Path, str]]:
    matches = []
    needle = name_contains.casefold()
    for name_path in sorted(sys_input_root.glob("event*/device/name")):
        try:
            name = name_path.read_text().strip()
        except OSError:
            continue
        if needle in name.casefold():
            matches.append((dev_input_root / name_path.parents[1].name, name))
    return matches


def _reconnect_unbound_usb_hid(
    name_contains: str,
    *,
    sys_usb_root: Path = Path("/sys/bus/usb/devices"),
    dev_usb_root: Path = Path("/dev/bus/usb"),
) -> bool:
    """Ask usbfs to reprobe matching, currently unbound HID interfaces.

    A libusb process can detach ``usbhid`` and then exit without reattaching
    it.  USBDEVFS_CONNECT performs the same reconnect without requiring write
    access to ``/sys/bus/usb/drivers/usbhid/bind``.  If another process still
    owns the interface, the kernel rejects this request instead of stealing
    the device.
    """

    needle = name_contains.casefold()
    reconnected = False
    for product_path in sorted(sys_usb_root.glob("*/product")):
        try:
            product = product_path.read_text().strip()
        except OSError:
            continue
        if needle not in product.casefold():
            continue

        usb_device_dir = product_path.parent
        for interface_dir in sorted(sys_usb_root.glob(f"{usb_device_dir.name}:*")):
            try:
                interface_class = int(
                    (interface_dir / "bInterfaceClass").read_text().strip(), 16
                )
                interface_number = int(
                    (interface_dir / "bInterfaceNumber").read_text().strip(), 16
                )
            except (OSError, ValueError):
                continue
            if interface_class != 0x03 or (interface_dir / "driver").exists():
                continue

            try:
                bus_number = int((usb_device_dir / "busnum").read_text().strip())
                device_number = int((usb_device_dir / "devnum").read_text().strip())
            except (OSError, ValueError):
                continue
            usb_path = dev_usb_root / f"{bus_number:03d}" / f"{device_number:03d}"
            try:
                fd = os.open(usb_path, os.O_RDWR | os.O_NONBLOCK)
            except OSError as exc:
                logger.debug("Cannot open %s to reconnect %s: %s", usb_path, product, exc)
                continue
            try:
                request = USBDEVFS_IOCTL_ARGUMENT.pack(
                    interface_number,
                    USBDEVFS_CONNECT,
                    0,
                )
                fcntl.ioctl(fd, USBDEVFS_IOCTL, request)
            except OSError as exc:
                logger.debug("Cannot reconnect %s at %s: %s", product, usb_path, exc)
            else:
                logger.info(
                    "Requested kernel HID reconnect for %s at %s interface %d",
                    product,
                    usb_path,
                    interface_number,
                )
                reconnected = True
            finally:
                os.close(fd)
    return reconnected


def discover_spacemouse(name_contains: str) -> Path:
    matches = _matching_event_devices(name_contains)
    if not matches and _reconnect_unbound_usb_hid(name_contains):
        # Driver probing creates the sysfs event before udev has necessarily
        # applied the device-node ACL.  Wait until the matching node is
        # present and readable, otherwise the immediate open can race udev.
        for _ in range(20):
            matches = _matching_event_devices(name_contains)
            if matches and all(
                path.exists() and os.access(path, os.R_OK) for path, _name in matches
            ):
                break
            time.sleep(0.05)
    if not matches:
        raise FileNotFoundError(
            f"no Linux input event device contains {name_contains!r}; "
            "ensure no libusb collector owns the device, or set "
            "spacemouse.device_path explicitly if auto-discovery is unsuitable"
        )
    if len(matches) > 1:
        details = ", ".join(f"{path} ({name})" for path, name in matches)
        raise RuntimeError(f"multiple SpaceMouse devices matched; set device_path: {details}")
    return matches[0][0]


class SpaceMouseReader:
    """Read the kernel evdev interface without an optional Python HID dependency."""

    def __init__(self, config: SpaceMouseConfig):
        self.config = config
        self.device_path: Path | None = None
        self._fd: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._axis_ranges: dict[int, tuple[int, int]] = {}
        self._working_axes = [0.0] * 6
        self._working_buttons: set[int] = set()
        self._state = SpaceMouseState()
        self._error: BaseException | None = None

    def open(self) -> None:
        if self._fd is not None:
            return
        path = (
            Path(self.config.device_path).expanduser()
            if self.config.device_path is not None
            else discover_spacemouse(self.config.name_contains)
        )
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            for code in ABS_CODES:
                current, minimum, maximum, _fuzz, _flat, _resolution = ABS_INFO.unpack(
                    fcntl.ioctl(fd, _ev_iocgabs(code), bytes(ABS_INFO.size))
                )
                if minimum >= maximum:
                    raise RuntimeError(
                        f"SpaceMouse axis {code} has an invalid range [{minimum}, {maximum}]"
                    )
                self._axis_ranges[code] = (minimum, maximum)
                self._working_axes[code] = self._normalize_axis(current, minimum, maximum)
        except BaseException:
            os.close(fd)
            raise

        self.device_path = path
        self._fd = fd
        self._error = None
        self._stop.clear()
        self._publish()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="spacemouse-evdev",
            daemon=True,
        )
        self._thread.start()
        logger.info("SpaceMouse opened: %s", path)

    def close(self) -> None:
        self._stop.set()
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            self._state = SpaceMouseState(received_monotonic_s=time.monotonic())

    def snapshot(self) -> SpaceMouseState:
        self._raise_if_failed()
        with self._lock:
            return self._state

    @staticmethod
    def _normalize_axis(value: int, minimum: int, maximum: int) -> float:
        scale = maximum if value >= 0 else -minimum
        if scale <= 0:
            return 0.0
        return max(-1.0, min(1.0, float(value) / float(scale)))

    def _read_loop(self) -> None:
        assert self._fd is not None
        fd = self._fd
        pending = b""
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select([fd], [], [], 0.1)
                if not readable:
                    continue
                chunk = os.read(fd, INPUT_EVENT.size * 64)
                if not chunk:
                    raise OSError(errno.ENODEV, "SpaceMouse disconnected")
                pending += chunk
                while len(pending) >= INPUT_EVENT.size:
                    event, pending = pending[: INPUT_EVENT.size], pending[INPUT_EVENT.size :]
                    _sec, _usec, event_type, code, value = INPUT_EVENT.unpack(event)
                    self._process_event(event_type, code, value)
        except OSError as exc:
            if not self._stop.is_set() and exc.errno not in {errno.EBADF, errno.EINTR}:
                self._error = exc
                logger.exception("SpaceMouse input failed")
        except BaseException as exc:
            if not self._stop.is_set():
                self._error = exc
                logger.exception("SpaceMouse input failed")

    def _process_event(self, event_type: int, code: int, value: int) -> None:
        if event_type == EV_ABS and code in self._axis_ranges:
            minimum, maximum = self._axis_ranges[code]
            self._working_axes[code] = self._normalize_axis(value, minimum, maximum)
        elif event_type == EV_KEY and code >= BTN_MISC:
            index = code - BTN_MISC
            if value:
                self._working_buttons.add(index)
            else:
                self._working_buttons.discard(index)
        elif event_type == EV_SYN and code == SYN_REPORT:
            self._publish()

    def _publish(self) -> None:
        state = SpaceMouseState(
            translation=tuple(self._working_axes[:3]),
            rotation=tuple(self._working_axes[3:]),
            buttons=frozenset(self._working_buttons),
            received_monotonic_s=time.monotonic(),
            connected=True,
        )
        with self._lock:
            self._state = state

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"SpaceMouse input failed: {self._error}") from self._error
