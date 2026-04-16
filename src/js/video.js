const VIDEO_PLAYBACK_RATE = 0.7;
const VIDEO_FEEDBACK_FLASH_MS = 880;
/** Same slop as qualitative triplet: tap vs drag. */
const VID_PC_TAP_MOVE_THRESHOLD_PX = 12;

function getPlaybackFeedbackEl(btn) {
  if (!btn || !btn.getAttribute) return null;
  const id = btn.getAttribute('data-feedback-id');
  return id ? document.getElementById(id) : null;
}

window.flashVideoPlaybackFeedback = function flashVideoPlaybackFeedback(feedbackEl, mode) {
  if (!feedbackEl) return;
  window.clearTimeout(feedbackEl._playbackFeedbackTimer);
  feedbackEl.classList.remove('is-visible', 'is-pause', 'is-play');
  void feedbackEl.offsetWidth;
  feedbackEl.classList.add('is-visible', mode === 'pause' ? 'is-pause' : 'is-play');
  feedbackEl._playbackFeedbackTimer = window.setTimeout(() => {
    feedbackEl.classList.remove('is-visible', 'is-pause', 'is-play');
    feedbackEl._playbackFeedbackTimer = 0;
  }, VIDEO_FEEDBACK_FLASH_MS);
};

function syncPlaybackToggleButton(video, btn) {
  if (!video || !btn) return;
  const text = btn.querySelector('.qualitative-video-playback-toggle__text');
  const paused = video.paused;
  if (text) text.textContent = paused ? 'Play' : 'Pause';
  btn.setAttribute('aria-pressed', paused ? 'true' : 'false');
  btn.setAttribute('aria-label', paused ? 'Play video' : 'Pause video');
}

window.toggleVideo = function toggleVideo(id, btn) {
  const v = document.getElementById(id);
  if (!v) return;
  const feedbackEl = getPlaybackFeedbackEl(btn);

  if (v.paused) {
    if (feedbackEl) flashVideoPlaybackFeedback(feedbackEl, 'play');
    v.play().catch(() => {});
  } else {
    if (feedbackEl) flashVideoPlaybackFeedback(feedbackEl, 'pause');
    v.pause();
  }
};

document.querySelectorAll('video').forEach((v) => {
  function setRate() {
    v.playbackRate = VIDEO_PLAYBACK_RATE;
  }
  v.addEventListener('loadedmetadata', setRate);
  if (v.readyState >= 1) setRate();
});

function bindVidPcVideoTapToggle() {
  const frame = document.querySelector('#geometry .geometry-single-video-frame');
  const btn = document.getElementById('vid_pc-playback-btn');
  if (!frame || !btn) return;
  let tapDown = null;

  frame.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    if (e.target.closest('.qualitative-video-playback-toggle')) return;
    tapDown = { x: e.clientX, y: e.clientY };
  }, { passive: true });

  frame.addEventListener('pointerup', (e) => {
    if (!tapDown) return;
    const dx = Math.abs(e.clientX - tapDown.x);
    const dy = Math.abs(e.clientY - tapDown.y);
    if (dx > VID_PC_TAP_MOVE_THRESHOLD_PX || dy > VID_PC_TAP_MOVE_THRESHOLD_PX) tapDown = null;
  }, { passive: true });

  frame.addEventListener('click', (e) => {
    if (e.target.closest('.qualitative-video-playback-toggle')) return;
    if (!tapDown) return;
    const dx = Math.abs(e.clientX - tapDown.x);
    const dy = Math.abs(e.clientY - tapDown.y);
    tapDown = null;
    if (dx > VID_PC_TAP_MOVE_THRESHOLD_PX || dy > VID_PC_TAP_MOVE_THRESHOLD_PX) return;
    e.preventDefault();
    toggleVideo('vid_pc', btn);
  });

  frame.addEventListener('pointercancel', () => { tapDown = null; }, { passive: true });
}

document.addEventListener('DOMContentLoaded', () => {
  const v = document.getElementById('vid_pc');
  const btn = document.getElementById('vid_pc-playback-btn');
  if (v && btn) {
    const sync = () => syncPlaybackToggleButton(v, btn);
    v.addEventListener('play', sync);
    v.addEventListener('pause', sync);
    sync();
  }
  bindVidPcVideoTapToggle();
});
