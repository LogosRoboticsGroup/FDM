(() => {
  const status = document.querySelector('#status');
  const vrStatus = document.querySelector('#vr-status');
  const inputStatus = document.querySelector('#input-status');
  const scene = document.querySelector('a-scene');
  const controllers = { left: document.querySelector('#left'), right: document.querySelector('#right') };
  const buttonEvents = {
    left: { trigger: new Set(), squeeze: new Set() },
    right: { trigger: new Set(), squeeze: new Set() },
  };
  const recordingButtons = {
    x: false,
    a: false,
    b: false,
    sampledX: false,
    pendingEvent: null,
  };
  const boundSessions = new WeakSet();
  let sessionId = null;
  let socket;
  let seq = 0;

  function setConnectionStatus(value) {
    status.textContent = value;
    vrStatus?.setAttribute('value', `Network: ${value.toLowerCase()}`);
  }

  function connect() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${scheme}://${location.hostname}:${window.STARVLA_VR.websocketPort}/vr`);
    socket.onopen = () => setConnectionStatus('Connected');
    socket.onclose = () => {
      recordingButtons.pendingEvent = null;
      setConnectionStatus('Disconnected');
      setTimeout(connect, 1000);
    };
    socket.onerror = () => socket.close();
  }

  function bindInputEvents(side, entity) {
    const bind = (down, up, name) => {
      entity.addEventListener(down, () => buttonEvents[side][name].add(name));
      entity.addEventListener(up, () => buttonEvents[side][name].delete(name));
    };
    bind('triggerdown', 'triggerup', 'trigger');
    bind('gripdown', 'gripup', 'squeeze');
    const bindRecording = (down, up, name, event) => {
      entity.addEventListener(down, () => {
        if (!recordingButtons[name]) {
          recordingButtons.pendingEvent = recordingButtons.pendingEvent || event;
        }
        recordingButtons[name] = true;
      });
      entity.addEventListener(up, () => { recordingButtons[name] = false; });
    };
    if (side === 'left') {
      bindRecording('xbuttondown', 'xbuttonup', 'x', 'start_episode');
    }
    if (side === 'right') {
      bindRecording('abuttondown', 'abuttonup', 'a', 'save_episode_and_home');
      bindRecording('bbuttondown', 'bbuttonup', 'b', 'discard_episode_and_home');
    }
  }

  function currentXrSession() {
    return scene.xrSession || scene.renderer?.xr?.getSession();
  }

  function bindXrSession(session) {
    if (!session || boundSessions.has(session)) return;
    boundSessions.add(session);
    const bind = (down, up, name) => {
      session.addEventListener(down, (event_) => {
        const side = event_.inputSource?.handedness;
        if (side in buttonEvents) buttonEvents[side][name].add(event_.inputSource);
      });
      session.addEventListener(up, (event_) => {
        const side = event_.inputSource?.handedness;
        if (side in buttonEvents) buttonEvents[side][name].delete(event_.inputSource);
      });
    };
    bind('selectstart', 'selectend', 'trigger');
    bind('squeezestart', 'squeezeend', 'squeeze');
    session.addEventListener('end', () => {
      Object.values(buttonEvents).forEach((buttons) => {
        buttons.trigger.clear();
        buttons.squeeze.clear();
      });
      recordingButtons.x = false;
      recordingButtons.a = false;
      recordingButtons.b = false;
      recordingButtons.sampledX = false;
      recordingButtons.pendingEvent = null;
    }, { once: true });
  }

  function inputSource(side) {
    return Array.from(currentXrSession()?.inputSources || [])
      .find((source) => source.handedness === side && !source.hand) || null;
  }

  function buttonValue(button) {
    if (!button) return 0;
    const value = Number.isFinite(Number(button.value)) ? Number(button.value) : 0;
    return Math.max(value, button.pressed ? 1 : 0);
  }

  function controllerButtonValue(gamepad, index, eventFallback) {
    const button = gamepad?.buttons?.[index];
    // A gamepad button is the authoritative sampled state. Controller events
    // are only a fallback for runtimes that do not expose gamepad buttons;
    // otherwise one missed "up" event can leave the Set stuck and falsely
    // report a trigger as pressed for an entire episode.
    return button ? buttonValue(button) : (eventFallback.size ? 1 : 0);
  }

  function controllerGamepad(side, entity) {
    const source = inputSource(side);
    const controller = entity.components['tracked-controls']?.controller;
    return source?.gamepad || controller?.gamepad || controller || null;
  }

  function handFrame(side, entity) {
    const controller = entity.components['tracked-controls']?.controller || null;
    const source = inputSource(side) || controller;
    const xrFrame = scene.frame;
    const referenceSpace = scene.renderer?.xr?.getReferenceSpace();
    const poseSpace = source?.gripSpace || source?.targetRaySpace;
    const pose = xrFrame && referenceSpace && poseSpace
      ? xrFrame.getPose(poseSpace, referenceSpace)
      : null;
    if (!pose) return { tracked: false };
    const position = pose.transform.position;
    const quaternion = pose.transform.orientation;
    const gamepad = controllerGamepad(side, entity);
    return {
      tracked: true,
      position_m: [position.x, position.y, position.z],
      quaternion_xyzw: [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
      trigger: controllerButtonValue(gamepad, 0, buttonEvents[side].trigger),
      squeeze: controllerButtonValue(gamepad, 1, buttonEvents[side].squeeze),
    };
  }

  function sendEvent(event) {
    if (!sessionId || socket?.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({ type: 'event', event }));
  }

  function sendFrame() {
    bindXrSession(currentXrSession());
    const hands = {
      left: handFrame('left', controllers.left),
      right: handFrame('right', controllers.right),
    };
    // Meta Quest Touch profiles expose X at gamepad button index 4. Keep the
    // A-Frame xbuttondown path above as the primary semantic event and use this
    // sampled edge as a fallback for runtimes that do not emit that event.
    const sampledX = buttonValue(
      controllerGamepad('left', controllers.left)?.buttons?.[4],
    ) >= 0.5;
    if (sampledX && !recordingButtons.sampledX) {
      recordingButtons.pendingEvent = recordingButtons.pendingEvent || 'start_episode';
    }
    recordingButtons.sampledX = sampledX;
    const record = {
      xPressed: recordingButtons.x || sampledX,
      aPressed: recordingButtons.a,
      bPressed: recordingButtons.b,
      event: recordingButtons.pendingEvent,
    };
    inputStatus?.setAttribute(
      'value',
      `L trigger: ${(hands.left.trigger || 0).toFixed(2)}  squeeze: ${(hands.left.squeeze || 0).toFixed(2)}\n`
        + `R trigger: ${(hands.right.trigger || 0).toFixed(2)}  squeeze: ${(hands.right.squeeze || 0).toFixed(2)}\n`
        + `X start: ${record.xPressed ? 'DOWN' : 'up'}  `
        + `A stop/home/save: ${record.aPressed ? 'DOWN' : 'up'}  `
        + `B stop/home/discard: ${record.bPressed ? 'DOWN' : 'up'}`,
    );
    let frameSent = false;
    if (
      sessionId
      && socket?.readyState === WebSocket.OPEN
      && (socket.bufferedAmount === 0 || record.event)
    ) {
      socket.send(JSON.stringify({
        protocol: 'starvla.vr', version: 1, session_id: sessionId,
        seq: seq++, client_timestamp_ms: performance.timeOrigin + performance.now(), hands,
      }));
      frameSent = true;
    }
    if (record.event && frameSent) {
      sendEvent(record.event);
      recordingButtons.pendingEvent = null;
    }
  }

  AFRAME.registerComponent('teleop-frame-pump', { tick: sendFrame });
  scene.setAttribute('teleop-frame-pump', '');
  Object.entries(controllers).forEach(([side, entity]) => bindInputEvents(side, entity));
  scene.addEventListener('enter-vr', () => {
    sessionId = crypto.randomUUID();
    seq = 0;
    bindXrSession(currentXrSession());
  });
  scene.addEventListener('exit-vr', () => { sessionId = null; });
  document.querySelector('#enter-vr').onclick = () => scene.enterVR();
  connect();
})();
