// Interactive comparison slider (supports multiple blocks + JSON-configured scenes)

function getInlineComparisonData(jsonUrl) {
  if (!jsonUrl || !document || !document.querySelector) return null;
  const el = document.querySelector(`script[data-inline-json="${CSS.escape(jsonUrl)}"]`);
  if (!el) return null;
  try {
    return JSON.parse(el.textContent || 'null');
  } catch (_err) {
    return null;
  }
}

function initComparisonSlider() {
  const roots = document.querySelectorAll('[data-comparison-root]');
  if (!roots.length) return;
  roots.forEach(function (root) {
    initComparisonSliderBlock(root);
  });
}

function initComparisonSliderBlock(root) {
  if (!root || root.__comparisonSliderInitialized) return;
  root.__comparisonSliderInitialized = true;

  const slider = root.querySelector('[data-comparison="slider"]');
  const range = root.querySelector('[data-comparison="range"]');
  const leftVideo = root.querySelector('[data-comparison="leftVideo"]');
  const rightVideo = root.querySelector('[data-comparison="rightVideo"]');
  const sceneTitle = root.querySelector('[data-comparison="sceneTitle"]');
  const subtitle = root.querySelector('[data-comparison="subtitle"]');
  const caption = root.closest('.results-block')?.querySelector('[data-comparison="caption"]') || root.querySelector('[data-comparison="caption"]');
  const leftLabel = root.querySelector('[data-comparison="leftLabel"]');
  const rightLabel = root.querySelector('[data-comparison="rightLabel"]');
  const dotsContainer = root.querySelector('[data-comparison="dots"]');
  const prevBtn = root.querySelector('[data-comparison-action="prev"]');
  const nextBtn = root.querySelector('[data-comparison-action="next"]');
  const pauseBtn = root.querySelector('[data-comparison-action="togglePause"]');
  const cell = root.closest('.method-cell');

  if (!slider || !range || !leftVideo || !rightVideo) return;

  const isGeometry = root.classList.contains('comparison-wrap--geometry');
  const geometryPlaybackBtn = root.querySelector('[data-geometry-playback-toggle]');
  const geometryFeedbackEl = root.querySelector('#geometry-comparison-playback-feedback');

  let currentSceneIndex = 0;
  let currentSplit = Number(range.value) || 50;
  let scenes = [];
  let defaults = {};
  let isPaused = false;
  let sceneLoadToken = 0;

  function updatePauseButton() {
    if (!pauseBtn) return;
    if (isPaused) {
      pauseBtn.setAttribute('aria-label', 'Continue playback');
      pauseBtn.setAttribute('title', 'Continue');
      pauseBtn.innerHTML = '<i class="fas fa-play" aria-hidden="true"></i>';
    } else {
      pauseBtn.setAttribute('aria-label', 'Pause playback');
      pauseBtn.setAttribute('title', 'Pause');
      pauseBtn.innerHTML = '<i class="fas fa-pause" aria-hidden="true"></i>';
    }
  }

  function syncGeometryPlaybackButton() {
    if (!geometryPlaybackBtn) return;
    const text = geometryPlaybackBtn.querySelector('.qualitative-video-playback-toggle__text');
    if (text) text.textContent = isPaused ? 'Play' : 'Pause';
    geometryPlaybackBtn.setAttribute('aria-pressed', isPaused ? 'true' : 'false');
    geometryPlaybackBtn.setAttribute(
      'aria-label',
      isPaused ? 'Play comparison videos' : 'Pause comparison videos',
    );
  }

  function toggleGeometryPlaybackWithFeedback() {
    setPausedState(!isPaused);
    if (geometryFeedbackEl && window.flashVideoPlaybackFeedback) {
      window.flashVideoPlaybackFeedback(geometryFeedbackEl, isPaused ? 'pause' : 'play');
    }
  }

  function setPausedState(shouldPause) {
    isPaused = !!shouldPause;
    updatePauseButton();
    if (isGeometry) syncGeometryPlaybackButton();

    if (isPaused) {
      leftVideo.pause();
      rightVideo.pause();
    } else {
      leftVideo.play().catch(function () { });
      rightVideo.play().catch(function () { });
    }
  }

  function normalizeScene(scene) {
    const s = scene && typeof scene === 'object' ? scene : {};
    return {
      sceneTitle: s.sceneTitle ?? s.title ?? s.name ?? '',
      subtitle: s.subtitle ?? '',
      caption: s.caption ?? '',
      leftLabel: s.leftLabel ?? '',
      rightLabel: s.rightLabel ?? '',
      leftIsOurs: !!s.leftIsOurs,
      rightIsOurs: !!s.rightIsOurs,
      leftVideo: s.leftVideo ?? s.left ?? '',
      rightVideo: s.rightVideo ?? s.right ?? '',
    };
  }

  function mergeSceneWithDefaults(scene) {
    const s = normalizeScene(scene);
    const d = defaults && typeof defaults === 'object' ? defaults : {};
    return {
      ...s,
      sceneTitle: s.sceneTitle ?? '',
      subtitle: s.subtitle || d.subtitle || '',
      caption: s.caption || d.caption || '',
      leftLabel: s.leftLabel || d.leftLabel || '',
      rightLabel: s.rightLabel || d.rightLabel || '',
      leftIsOurs: typeof scene?.leftIsOurs === 'boolean' ? scene.leftIsOurs : !!d.leftIsOurs,
      rightIsOurs: typeof scene?.rightIsOurs === 'boolean' ? scene.rightIsOurs : !!d.rightIsOurs,
      leftVideo: s.leftVideo || d.leftVideo || '',
      rightVideo: s.rightVideo || d.rightVideo || '',
    };
  }

  function setLoadingState(isLoaded) {
    if (!cell) return;
    cell.classList.toggle('is-video-loaded', !!isLoaded);
  }

  function scopeLoadingOverlayToSlider() {
    if (!cell || !slider) return;
    const loader = cell.querySelector('.video-loading');
    if (!loader) return;
    if (loader.parentElement === slider) return;
    slider.appendChild(loader);
  }

  function setVisualSplit(value) {
    const clamped = Math.min(100, Math.max(0, value));
    slider.style.setProperty('--split', clamped + '%');
    return clamped;
  }

  function applySplit(value) {
    currentSplit = setVisualSplit(value);
  }

  function ratioFromClientX(clientX) {
    const rect = slider.getBoundingClientRect();
    if (rect.width <= 0) return currentSplit;
    return Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
  }

  function readVideoAspectRatioString(video) {
    if (!video) return null;
    const w = Number(video.videoWidth) || 0;
    const h = Number(video.videoHeight) || 0;
    if (w <= 0 || h <= 0) return null;
    // Use `w / h` syntax for maximum CSS compatibility.
    return w + ' / ' + h;
  }

  function applySliderAspectRatioFromLoadedVideos() {
    // Prefer the left/overlay video first.
    // This matches the UX expectation for overlay comparisons, and avoids
    // situations where the "base" video's metadata differs from what you
    // perceive visually.
    const ratioStr = readVideoAspectRatioString(leftVideo) ?? readVideoAspectRatioString(rightVideo);
    if (!ratioStr) return;
    slider.style.setProperty('aspect-ratio', ratioStr);
  }

  function setVideoSrc(video, src) {
    const source = video.querySelector('source');
    if (source) source.src = src || '';
    // Also set `src` for browsers that use it directly.
    video.removeAttribute('src');
    video.load();
  }

  function waitForSceneVideosToSettle(loadToken, onSettled) {
    let leftDone = false;
    let rightDone = false;
    let settled = false;

    setLoadingState(false);

    function maybeDone() {
      if (settled) return;
      if (leftDone && rightDone) {
        settled = true;
        // Ignore stale completions from a previous scene request.
        if (loadToken !== sceneLoadToken) return;
        applySliderAspectRatioFromLoadedVideos();
        setLoadingState(true);
        if (typeof onSettled === 'function') onSettled();
      }
    }

    function markLeftDone() {
      leftDone = true;
      maybeDone();
    }
    function markRightDone() {
      rightDone = true;
      maybeDone();
    }

    // Stricter gate for synchronized starts:
    // readyState >= 3 (HAVE_FUTURE_DATA) approximates `canplay`.
    if (leftVideo.readyState >= 3) markLeftDone();
    if (rightVideo.readyState >= 3) markRightDone();

    const opts = { once: true };
    leftVideo.addEventListener('canplay', markLeftDone, opts);
    leftVideo.addEventListener('error', markLeftDone, opts);

    rightVideo.addEventListener('canplay', markRightDone, opts);
    rightVideo.addEventListener('error', markRightDone, opts);

    // Safety: don't spin forever if the browser never fires events.
    window.setTimeout(function () {
      if (settled) return;
      settled = true;
      // Ignore stale completions from a previous scene request.
      if (loadToken !== sceneLoadToken) return;
      applySliderAspectRatioFromLoadedVideos();
      setLoadingState(true);
      if (typeof onSettled === 'function') onSettled();
    }, 2500);
  }

  function playBothFromStart() {
    // Wait for both videos to finish seeking to 0 before issuing play()
    // so the browser starts decoding both from the same point simultaneously.
    function seekAndReady(video) {
      return new Promise(function (resolve) {
        if (video.currentTime === 0 && video.readyState >= 2) {
          resolve();
          return;
        }
        function onSeeked() {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        }
        video.addEventListener('seeked', onSeeked);
        try { video.currentTime = 0; } catch { resolve(); }
      });
    }
    return Promise.all([seekAndReady(leftVideo), seekAndReady(rightVideo)]).then(function () {
      const p1 = leftVideo.play().catch(function () { });
      const p2 = rightVideo.play().catch(function () { });
      return Promise.allSettled([p1, p2]);
    });
  }

  function syncPlaybackEvents() {
    // Mirror play/pause bidirectionally so either video can drive state.
    function mirrorPlayPause(a, b) {
      a.addEventListener('play', function () {
        if (b.paused) b.play().catch(function () { });
      });
      a.addEventListener('pause', function () {
        if (!b.paused) b.pause();
      });
    }
    mirrorPlayPause(leftVideo, rightVideo);
    mirrorPlayPause(rightVideo, leftVideo);

    // Drift correction is one-directional: leftVideo is master, rightVideo is slave.
    // A 0.15 s threshold prevents constant seeks (which themselves cause drift/stutter).
    leftVideo.addEventListener('timeupdate', function () {
      if (leftVideo.paused) return;
      try {
        if (Math.abs(leftVideo.currentTime - rightVideo.currentTime) > 0.15) {
          rightVideo.currentTime = leftVideo.currentTime;
        }
      } catch { }
    });
  }

  function buildDots() {
    if (!dotsContainer) return;
    dotsContainer.innerHTML = '';
    scenes.forEach(function (_scene, idx) {
      const dot = document.createElement('button');
      dot.type = 'button';
      dot.className = 'scene-dot' + (idx === currentSceneIndex ? ' is-active' : '');
      dot.setAttribute('aria-label', 'Go to scene ' + (idx + 1));
      dot.addEventListener('click', function () {
        goToScene(idx);
      });
      dotsContainer.appendChild(dot);
    });
  }

  function updateDotsActive() {
    if (!dotsContainer) return;
    const dots = dotsContainer.querySelectorAll('.scene-dot');
    dots.forEach(function (dot, idx) {
      dot.classList.toggle('is-active', idx === currentSceneIndex);
    });
  }

  function loadScene(index) {
    if (!scenes.length) {
      if (sceneTitle) sceneTitle.textContent = '';
      if (subtitle) subtitle.textContent = '';
      if (caption) caption.textContent = '';
      if (leftLabel) leftLabel.textContent = '';
      if (rightLabel) rightLabel.textContent = '';
      setVideoSrc(leftVideo, '');
      setVideoSrc(rightVideo, '');
      if (isPaused) {
        leftVideo.pause();
        rightVideo.pause();
      }
      if (dotsContainer) dotsContainer.innerHTML = '';
      setLoadingState(true);
      return;
    }

    const clampedIndex = Math.max(0, Math.min(scenes.length - 1, index));
    currentSceneIndex = clampedIndex;
    const scene = mergeSceneWithDefaults(scenes[clampedIndex]);

    if (sceneTitle) sceneTitle.textContent = scene.sceneTitle || '';
    if (subtitle) subtitle.textContent = scene.subtitle || '';
    if (caption) caption.textContent = scene.caption || '';

    if (leftLabel) {
      leftLabel.textContent = scene.leftLabel || '';
      leftLabel.classList.toggle('comparison-corner-label-ours', !!scene.leftIsOurs);
    }
    if (rightLabel) {
      rightLabel.textContent = scene.rightLabel || '';
      rightLabel.classList.toggle('comparison-corner-label-ours', !!scene.rightIsOurs);
    }

    const loadToken = ++sceneLoadToken;
    setVideoSrc(leftVideo, scene.leftVideo);
    setVideoSrc(rightVideo, scene.rightVideo);
    waitForSceneVideosToSettle(loadToken, function () {
      if (!isPaused) playBothFromStart();
    });

    // Prevent autoplay races when the slider is currently paused.
    if (isPaused) {
      leftVideo.pause();
      rightVideo.pause();
    }

    updateDotsActive();
  }

  function goToScene(idx) {
    loadScene(idx);
  }

  function prevScene() {
    if (!scenes.length) return;
    const nextIndex = (currentSceneIndex - 1 + scenes.length) % scenes.length;
    goToScene(nextIndex);
  }

  function nextScene() {
    if (!scenes.length) return;
    const nextIndex = (currentSceneIndex + 1) % scenes.length;
    goToScene(nextIndex);
  }

  // Split UX (kept local to this instance)
  scopeLoadingOverlayToSlider();
  applySplit(currentSplit);

  range.addEventListener('input', function () {
    const value = Number(range.value) || 0;
    applySplit(value);
  });

  function updateFromClientX(clientX) {
    const value = ratioFromClientX(clientX);
    range.value = String(value);
    applySplit(value);
  }

  let isDragging = false;
  const DRAG_SLOP_PX = 6;

  if (isGeometry) {
    let pePointerId = null;
    let peDownX = 0;
    let peStartedOnHandle = false;
    let peHasDragged = false;

    function geometryPointerDown(e) {
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      pePointerId = e.pointerId;
      peDownX = e.clientX;
      peStartedOnHandle = !!(e.target && e.target.closest && e.target.closest('.comparison-handle'));
      peHasDragged = false;
      isDragging = false;
    }

    function geometryPointerMove(e) {
      if (pePointerId !== e.pointerId) return;
      if (peStartedOnHandle) {
        isDragging = true;
        peHasDragged = true;
        updateFromClientX(e.clientX);
        return;
      }
      if (Math.abs(e.clientX - peDownX) >= DRAG_SLOP_PX) {
        peHasDragged = true;
        isDragging = true;
      }
      if (isDragging) updateFromClientX(e.clientX);
    }

    function geometryPointerUp(e) {
      if (pePointerId !== e.pointerId) return;
      const dx = Math.abs(e.clientX - peDownX);
      if (!peStartedOnHandle && !peHasDragged && dx < DRAG_SLOP_PX) {
        if (!e.target.closest || !e.target.closest('.qualitative-video-playback-toggle')) {
          toggleGeometryPlaybackWithFeedback();
        }
      }
      pePointerId = null;
      peStartedOnHandle = false;
      peHasDragged = false;
      isDragging = false;
    }

    slider.addEventListener('pointerdown', geometryPointerDown);
    slider.addEventListener('pointermove', geometryPointerMove);
    window.addEventListener('pointerup', geometryPointerUp);
    slider.addEventListener('pointercancel', function () {
      pePointerId = null;
      isDragging = false;
    });

    if (geometryPlaybackBtn) {
      geometryPlaybackBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        toggleGeometryPlaybackWithFeedback();
      });
    }
  } else {
    slider.addEventListener('mousedown', function (e) {
      isDragging = true;
      updateFromClientX(e.clientX);
    });

    window.addEventListener('mousemove', function (e) {
      if (!isDragging) return;
      updateFromClientX(e.clientX);
    });

    window.addEventListener('mouseup', function () {
      isDragging = false;
    });

    slider.addEventListener(
      'touchstart',
      function (e) {
        if (!e.touches.length) return;
        isDragging = true;
        updateFromClientX(e.touches[0].clientX);
      },
      { passive: true },
    );

    window.addEventListener(
      'touchmove',
      function (e) {
        if (!isDragging || !e.touches.length) return;
        updateFromClientX(e.touches[0].clientX);
      },
      { passive: true },
    );

    window.addEventListener('touchend', function () {
      isDragging = false;
    });
  }

  if (prevBtn) prevBtn.addEventListener('click', prevScene);
  if (nextBtn) nextBtn.addEventListener('click', nextScene);
  if (pauseBtn) pauseBtn.addEventListener('click', function () { setPausedState(!isPaused); });
  updatePauseButton();
  if (isGeometry) syncGeometryPlaybackButton();

  syncPlaybackEvents();

  // Load scenes from JSON (optional). If missing/unreadable, we keep the block empty.
  const jsonUrl = root.getAttribute('data-comparison-scenes');
  if (!jsonUrl) {
    scenes = [];
    buildDots();
    loadScene(0);
    return;
  }

  const inlineData = getInlineComparisonData(jsonUrl);
  if (inlineData) {
    if (Array.isArray(inlineData)) {
      defaults = {};
      scenes = inlineData;
    } else {
      defaults = inlineData?.defaults ?? {};
      scenes = Array.isArray(inlineData?.scenes) ? inlineData.scenes : [];
    }
    buildDots();
    loadScene(0);
    return;
  }

  fetch(jsonUrl, { cache: 'no-store' })
    .then(function (res) {
      if (!res.ok) throw new Error('Failed to fetch scenes JSON: ' + res.status);
      return res.json();
    })
    .then(function (data) {
      if (Array.isArray(data)) {
        defaults = {};
        scenes = data;
      } else {
        defaults = data?.defaults ?? {};
        scenes = Array.isArray(data?.scenes) ? data.scenes : [];
      }
      buildDots();
      loadScene(0);
    })
    .catch(function () {
      defaults = {};
      scenes = [];
      buildDots();
      loadScene(0);
    });
}

