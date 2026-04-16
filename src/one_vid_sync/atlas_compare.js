(() => {
  if (window.__SITE_LITE__) return;

  const DRIFT_THRESHOLD_S = 0.05;
  const FEEDBACK_FLASH_MS = 850;
  const SPLIT_DRAG_SLOP_PX = 6;

  /**
   * Two-column merged asset (2× width): left strip = overlay (ours), right = RGB (baseline).
   * Same decoded file in both layers; horizontal split bar reveals more of the top (dots) vs bottom (render),
   * matching comparisonSlider geometry UX (not the 3-up triplet 2D hover split).
   */
  function initMergedDuo(cell) {
    const sliderSurface = cell.querySelector('.comparison-slider--merged-duo');
    const roleOurs = cell.querySelector('.sync-role--ours');
    const roleBase = cell.querySelector('.sync-role--baseline');
    const range = cell.querySelector('[data-merged-duo-range]');
    if (!sliderSurface || !roleOurs || !roleBase) return;

    const master = roleOurs.querySelector('video');
    const follower = roleBase.querySelector('video');
    if (!master || !follower) return;

    const allVideos = [master, follower];
    const methodCell = cell.closest('.method-cell');

    const sceneIndex = Number(cell.dataset.sceneIndex) || 0;
    const sceneCount = Number(cell.dataset.sceneCount) || 1;

    let currentSplit = 50;

    function applySplit(value) {
      const clamped = Math.min(100, Math.max(0, Number(value) || 0));
      currentSplit = clamped;
      sliderSurface.style.setProperty('--split', `${clamped}%`);
      if (range) range.value = String(Math.round(clamped));
    }

    function ratioFromClientX(clientX) {
      const rect = sliderSurface.getBoundingClientRect();
      if (rect.width <= 0) return currentSplit;
      return Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
    }

    function playAll() {
      allVideos.forEach((v) => v.play().catch(() => {}));
    }
    function pauseAll() {
      allVideos.forEach((v) => v.pause());
    }

    master.addEventListener('timeupdate', () => {
      if (master.paused) return;
      const t0 = master.currentTime;
      if (Math.abs(follower.currentTime - t0) > DRIFT_THRESHOLD_S) follower.currentTime = t0;
    });

    function setLoadedState() {
      if (methodCell) methodCell.classList.add('is-video-loaded');
    }

    function seekToScene() {
      if (!master.duration || !isFinite(master.duration)) return;
      if (master.videoWidth && master.videoHeight) {
        cell.style.aspectRatio = `${master.videoWidth / 2} / ${master.videoHeight}`;
      }
      const t = ((sceneIndex + 0.5) / sceneCount) * master.duration;
      allVideos.forEach((v) => {
        v.currentTime = t;
      });
      playAll();
    }

    function tryMarkLoaded() {
      if (master.readyState >= 3 && follower.readyState >= 3) setLoadedState();
    }

    applySplit(currentSplit);

    if (master.readyState >= 1 && isFinite(master.duration) && master.duration > 0) {
      seekToScene();
    } else {
      master.addEventListener('loadedmetadata', seekToScene, { once: true });
      master.addEventListener('canplay', seekToScene, { once: true });
    }
    master.addEventListener('canplay', tryMarkLoaded);
    follower.addEventListener('canplay', tryMarkLoaded);

    if (range) {
      range.addEventListener('input', () => applySplit(Number(range.value)));
    }

    let playBtn = cell.querySelector('.qualitative-video-playback-toggle');
    if (!playBtn) {
      playBtn = document.createElement('button');
      playBtn.type = 'button';
      playBtn.className = 'qualitative-video-playback-toggle';
      playBtn.innerHTML = '<span class="qualitative-video-playback-toggle__text"></span>';
      cell.appendChild(playBtn);
    }
    const btnText = playBtn.querySelector('.qualitative-video-playback-toggle__text') || playBtn;

    function syncBtn() {
      const playing = !master.paused;
      btnText.textContent = playing ? 'Pause' : 'Play';
      playBtn.setAttribute('aria-pressed', String(!playing));
    }
    master.addEventListener('play', syncBtn);
    master.addEventListener('pause', syncBtn);
    syncBtn();

    let feedbackEl = cell.querySelector('.qualitative-video-playback-feedback');
    if (!feedbackEl) {
      feedbackEl = document.createElement('div');
      feedbackEl.className = 'qualitative-video-playback-feedback';
      feedbackEl.setAttribute('aria-hidden', 'true');
      feedbackEl.innerHTML =
        '<div class="qualitative-video-playback-feedback__backdrop">' +
          '<svg class="qualitative-video-playback-feedback__icon qualitative-video-playback-feedback__icon--play" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="currentColor" d="M8 5v14l11-7z"/></svg>' +
          '<svg class="qualitative-video-playback-feedback__icon qualitative-video-playback-feedback__icon--pause" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>' +
        '</div>';
      cell.appendChild(feedbackEl);
    }
    let feedbackTimeout = 0;

    function flashFeedback(mode) {
      if (feedbackTimeout) {
        clearTimeout(feedbackTimeout);
        feedbackTimeout = 0;
      }
      feedbackEl.classList.remove('is-visible', 'is-play', 'is-pause');
      void feedbackEl.offsetWidth;
      feedbackEl.classList.add('is-visible', mode === 'pause' ? 'is-pause' : 'is-play');
      feedbackTimeout = window.setTimeout(() => {
        feedbackEl.classList.remove('is-visible', 'is-play', 'is-pause');
        feedbackTimeout = 0;
      }, FEEDBACK_FLASH_MS);
    }

    function togglePlayback() {
      const willPlay = master.paused;
      if (willPlay) playAll();
      else pauseAll();
      flashFeedback(willPlay ? 'play' : 'pause');
    }

    playBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      togglePlayback();
    });

    function updateFromClientX(clientX) {
      applySplit(ratioFromClientX(clientX));
    }

    let pePointerId = null;
    let peDownX = 0;
    let peStartedOnHandle = false;
    let peHasDragged = false;
    let isDragging = false;

    function duoPointerDown(e) {
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      if (e.target.closest && e.target.closest('.qualitative-video-playback-toggle')) return;
      pePointerId = e.pointerId;
      peDownX = e.clientX;
      peStartedOnHandle = !!(e.target && e.target.closest && e.target.closest('.comparison-handle'));
      peHasDragged = false;
      isDragging = false;
    }

    function duoPointerMove(e) {
      if (pePointerId !== e.pointerId) return;
      if (e.target.closest && e.target.closest('.qualitative-video-playback-toggle')) return;
      if (peStartedOnHandle) {
        isDragging = true;
        peHasDragged = true;
        updateFromClientX(e.clientX);
        return;
      }
      if (Math.abs(e.clientX - peDownX) >= SPLIT_DRAG_SLOP_PX) {
        peHasDragged = true;
        isDragging = true;
      }
      if (isDragging) updateFromClientX(e.clientX);
    }

    function duoPointerUp(e) {
      if (pePointerId !== e.pointerId) return;
      const dx = Math.abs(e.clientX - peDownX);
      if (!peStartedOnHandle && !peHasDragged && dx < SPLIT_DRAG_SLOP_PX) {
        if (!e.target.closest || !e.target.closest('.qualitative-video-playback-toggle')) {
          togglePlayback();
        }
      }
      pePointerId = null;
      peStartedOnHandle = false;
      peHasDragged = false;
      isDragging = false;
    }

    sliderSurface.addEventListener('pointerdown', duoPointerDown);
    sliderSurface.addEventListener('pointermove', duoPointerMove);
    window.addEventListener('pointerup', duoPointerUp);
    sliderSurface.addEventListener('pointercancel', () => {
      pePointerId = null;
      isDragging = false;
    });
  }

  // ── Tab switcher ─────────────────────────────────────────────────────── //
  function activateAtlasPanel(switcher, model) {
    switcher.querySelectorAll('[data-atlas-tab]').forEach((tab) => {
      const active = tab.dataset.atlasTab === model;
      tab.classList.toggle('is-active', active);
      tab.setAttribute('aria-selected', active ? 'true' : 'false');
    });
    switcher.querySelectorAll('[data-atlas-panel]').forEach((panel) => {
      panel.classList.toggle('is-hidden', panel.dataset.atlasPanel !== model);
    });
  }

  function init() {
    document.querySelectorAll('[data-merged-duo]').forEach(initMergedDuo);

    document.querySelectorAll('[data-atlas-switcher]').forEach((switcher) => {
      const tabs = Array.from(switcher.querySelectorAll('[data-atlas-tab]'));
      if (!tabs.length) return;
      tabs.forEach((tab) => {
        tab.addEventListener('click', () => activateAtlasPanel(switcher, tab.dataset.atlasTab));
      });
      const firstActive = tabs.find((t) => t.classList.contains('is-active')) || tabs[0];
      activateAtlasPanel(switcher, firstActive.dataset.atlasTab);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
