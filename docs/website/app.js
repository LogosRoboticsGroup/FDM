'use strict';

const demo = document.querySelector('#demo-video');
const demoLaunch = document.querySelector('#demo-launch');
demoLaunch.hidden = false;
document.querySelector('.demo-cover').hidden = false;
demo.controls = false;
demoLaunch.addEventListener('click', async () => {
  demoLaunch.hidden = true;
  demo.controls = true;
  demo.closest('.demo-stage').classList.add('playing');
  try {
    await demo.play();
  } catch {
    demoLaunch.hidden = false;
    demo.closest('.demo-stage').classList.remove('playing');
  }
});
demo.addEventListener('play', () => {
  demoLaunch.hidden = true;
  demo.closest('.demo-stage').classList.add('playing');
});

document.querySelectorAll('[data-seek]').forEach(button => {
  button.addEventListener('click', () => {
    demo.controls = true;
    demoLaunch.hidden = true;
    demo.closest('.demo-stage').classList.add('playing');
    demo.currentTime = Number(button.dataset.seek);
    demo.play().catch(() => { demoLaunch.hidden = false; });
  });
});

const modeCopy = {
  training: 'Train jointly with action imitation and future prediction. Asymmetric attention blocks future information from reaching actions, including through context.',
  inference: 'Keep the observation and instruction processing needed by the action expert. Skip future generation entirely; predicted futures are never an input to the policy.',
};
document.querySelectorAll('[data-mode]').forEach(button => {
  if (button.tagName !== 'BUTTON') return;
  button.addEventListener('click', () => {
    document.querySelectorAll('button[data-mode]').forEach(item => {
      const selected = item === button;
      item.classList.toggle('selected', selected);
      item.setAttribute('aria-pressed', String(selected));
    });
    document.querySelector('#pipeline-display').dataset.mode = button.dataset.mode;
    document.querySelector('#mode-explanation').textContent = modeCopy[button.dataset.mode];
  });
});

const tasks = {
  drawer: { name: 'Drawer packing', description: 'Place the objects in the drawer and close it.' },
  gift: { name: 'Box packing and closing', description: 'Pack the objects into the box and close the lid.' },
  badminton: { name: 'Shuttlecock insertion', description: 'Insert the shuttlecock into the holder.' },
  cup: { name: 'Paper-cup stacking', description: 'Pick up and stack the paper cups.' },
  cube: { name: 'Block stacking', description: 'Pick up the blocks and build a stack.' },
};
const methods = [
  { directory: 'FDM-VLM', name: 'FDM w/ VLM', ours: true },
  { directory: 'FDM-VGM', name: 'FDM w/ VGM', ours: true },
  { directory: 'pi0.5', name: 'π₀.₅', ours: false },
  { directory: 'FastWAM-joint', name: 'FastWAM-Joint', ours: false },
];
const grid = document.querySelector('#comparison-grid');
const speed = document.querySelector('#speed');
const playButton = document.querySelector('#play-all');
const status = document.querySelector('#playback-status');
let currentTask = 'drawer';
let generation = 0;
let groupPlaying = false;
let recordings = {};

function videos() {
  return [...grid.querySelectorAll('video')];
}

function pauseAll() {
  generation += 1;
  videos().forEach(video => video.pause());
  groupPlaying = false;
  playButton.textContent = 'Play all';
  playButton.disabled = false;
}

function renderComparisons() {
  pauseAll();
  grid.replaceChildren();
  document.querySelector('#task-description').textContent = tasks[currentTask].description;
  methods.forEach(method => {
    const key = `${currentTask}/${method.directory}.mp4`;
    const source = `assets/comparisons/${key}`;
    const article = document.createElement('article');
    article.className = 'comparison-card';
    const title = document.createElement('div');
    title.className = 'comparison-title';
    const heading = document.createElement('h3');
    heading.textContent = method.name;
    title.append(heading);
    if (method.ours) {
      const badge = document.createElement('span');
      badge.textContent = 'Ours';
      title.append(badge);
    }
    const video = document.createElement('video');
    video.controls = true;
    video.playsInline = true;
    video.muted = true;
    video.preload = 'metadata';
    video.poster = source.replace('.mp4', '.webp');
    video.src = source;
    video.playbackRate = Number(speed.value);
    video.setAttribute('aria-label', `${tasks[currentTask].name}, ${method.name}, third view above left and right wrist views`);
    const caption = document.createElement('div');
    caption.className = 'clip-caption';
    const duration = document.createElement('span');
    duration.dataset.key = key;
    duration.textContent = recordings[key] ? `${recordings[key].duration.toFixed(1)} s recording` : 'Original recording';
    const link = document.createElement('a');
    link.href = source;
    link.textContent = 'Open clip';
    caption.append(duration, link);
    video.addEventListener('loadedmetadata', () => {
      duration.textContent = `${video.duration.toFixed(1)} s recording`;
    });
    video.addEventListener('error', () => {
      status.textContent = `The ${method.name} clip could not load. Retry playback or use “Open clip”.`;
    });
    video.addEventListener('ended', () => {
      if (videos().every(item => item.ended)) {
        groupPlaying = false;
        playButton.textContent = 'Replay all';
        status.textContent = 'All recordings have finished. Replay or select another task.';
      }
    });
    article.append(title, video, caption);
    grid.append(article);
  });
  status.textContent = `${tasks[currentTask].name} · Three synchronized views · ${speed.selectedOptions[0].text} speed.`;
}

function ready(video) {
  if (video.readyState >= 3) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => finish(new Error('Loading timed out')), 25000);
    const loaded = () => finish();
    const failed = () => finish(new Error('Video unavailable'));
    function finish(error) {
      clearTimeout(timeout);
      video.removeEventListener('canplay', loaded);
      video.removeEventListener('error', failed);
      if (error) reject(error); else resolve();
    }
    video.addEventListener('canplay', loaded, { once: true });
    video.addEventListener('error', failed, { once: true });
    video.preload = 'auto';
    if (video.readyState === 0) video.load();
  });
}

playButton.addEventListener('click', async () => {
  if (groupPlaying) {
    pauseAll();
    status.textContent = 'Playback paused. Press Play all to continue.';
    return;
  }
  const token = ++generation;
  const current = videos();
  playButton.disabled = true;
  playButton.textContent = 'Loading…';
  status.textContent = `Loading ${current.length} available recordings…`;
  try {
    await Promise.all(current.map(ready));
    if (token !== generation) return;
    if (current.every(video => video.ended)) current.forEach(video => { video.currentTime = 0; });
    current.forEach(video => { video.playbackRate = Number(speed.value); });
    await Promise.all(current.filter(video => !video.ended).map(video => video.play()));
    if (token !== generation) {
      current.forEach(video => video.pause());
      return;
    }
    groupPlaying = true;
    playButton.textContent = 'Pause all';
    status.textContent = `Playing ${current.length} recordings at ${speed.value}×. Shorter clips hold their final frame.`;
  } catch {
    if (token !== generation) return;
    current.forEach(video => video.pause());
    playButton.textContent = 'Retry playback';
    status.textContent = 'Playback could not start. Retry or use the controls on each recording.';
  } finally {
    if (token === generation) playButton.disabled = false;
  }
});

document.querySelector('#restart-all').addEventListener('click', () => {
  pauseAll();
  videos().forEach(video => { if (video.readyState > 0) video.currentTime = 0; });
  status.textContent = 'Recordings reset to the beginning. Press Play all to start.';
});
document.querySelectorAll('[data-task]').forEach(button => {
  button.addEventListener('click', () => {
    currentTask = button.dataset.task;
    document.querySelectorAll('[data-task]').forEach(item => {
      item.classList.toggle('selected', item === button);
      item.setAttribute('aria-pressed', String(item === button));
    });
    renderComparisons();
  });
});
speed.addEventListener('change', () => {
  videos().forEach(video => { video.playbackRate = Number(speed.value); });
  status.textContent = `Playback speed set to ${speed.value}×.`;
});
renderComparisons();
fetch('assets/comparisons.json').then(response => {
  if (!response.ok) throw new Error('Duration metadata unavailable');
  return response.json();
}).then(data => {
  recordings = data;
  grid.querySelectorAll('[data-key]').forEach(item => {
    if (recordings[item.dataset.key]) item.textContent = `${recordings[item.dataset.key].duration.toFixed(1)} s recording`;
  });
}).catch(() => { /* Native video metadata remains the duration source. */ });

document.querySelector('#copy-citation').addEventListener('click', async () => {
  const citation = document.querySelector('#bibtex');
  const feedback = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    feedback.textContent = 'BibTeX copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    feedback.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
