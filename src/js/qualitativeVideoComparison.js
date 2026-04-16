(() => {
  if (window.__SITE_LITE__) return;

  const DEFAULT_SCENE_COUNT = 6;
  const SPLIT_DEFAULT_X = 0.5;
  const SPLIT_DEFAULT_Y = 0.5;

  const SYNC_HARD_SEEK_THRESHOLD_SECONDS = 0.012;
  const SYNC_THROTTLE_MS = 40;
  /** Eased return to center split when pointer leaves (ms); skipped if reduced motion. */
  const SPLIT_RESET_MS = 200;
  /** Max pointer movement (px) between down and click to count as a tap (not a split drag). */
  const TAP_TOGGLE_MOVE_THRESHOLD_PX = 12;
  const FEEDBACK_FLASH_MS = 880;

  function prefersReducedMotion() {
    try {
      return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch {
      return false;
    }
  }

  function clamp01(x) {
    if (!Number.isFinite(x)) return 0.5;
    return Math.min(1, Math.max(0, x));
  }

  function parseSceneCount(wrapper) {
    const n = Number(wrapper?.dataset?.sceneCount);
    return Number.isFinite(n) && n > 0 ? n : DEFAULT_SCENE_COUNT;
  }

  function getRoleVideo(tripletEl, role) {
    return tripletEl.querySelector(`video[data-sync-role="${CSS.escape(role)}"]`);
  }

  // Defers setSplitClip via rAF if the element has no layout dimensions yet.
  function setSplitClip(tripletEl, sx, sy) {
    const ours = tripletEl.__qualOurs;
    const baseline = tripletEl.__qualBaseline;
    const gt = tripletEl.__qualGt;
    if (!ours || !baseline || !gt) return;

    const rect = tripletEl.getBoundingClientRect();
    const w = tripletEl.clientWidth || rect.width || 0;
    const h = tripletEl.clientHeight || rect.height || 0;

    // If the element hasn't been laid out yet (e.g. panel just became visible),
    // retry after the browser has had a chance to paint.
    if (w <= 0 || h <= 0) {
      requestAnimationFrame(() => setSplitClip(tripletEl, sx, sy));
      return;
    }

    const cw = Math.round(sx * w);
    const ch = Math.round(sy * h);

    ours.style.clip      = `rect(0px, ${cw}px, ${ch}px, 0px)`;
    baseline.style.clip  = `rect(0px, ${w}px, ${ch}px, ${cw}px)`;
    gt.style.clip        = `rect(${ch}px, ${w}px, ${h}px, 0px)`;

    const xPct = (cw / w) * 100;
    const yPct = (ch / h) * 100;
    ours.style.clipPath      = `inset(0 ${100 - xPct}% ${100 - yPct}% 0)`;
    baseline.style.clipPath  = `inset(0 0 ${100 - yPct}% ${xPct}%)`;
    gt.style.clipPath        = `inset(${yPct}% 0 0 0)`;

    const frame1   = tripletEl.__frame1;
    const frame2   = tripletEl.__frame2;
    const frame3   = tripletEl.__frame3;
    const overlay  = tripletEl.__overlay;
    const label1   = tripletEl.__label1;
    const label2   = tripletEl.__label2;
    const label3   = tripletEl.__label3;

    if (overlay) {
      overlay.style.width  = `${w}px`;
      overlay.style.height = `${h}px`;
    }

    if (frame1) {
      frame1.style.left   = '0px';
      frame1.style.top    = '0px';
      frame1.style.width  = `${cw}px`;
      frame1.style.height = `${ch}px`;
    }
    if (frame2) {
      frame2.style.left   = `${cw}px`;
      frame2.style.top    = '0px';
      frame2.style.width  = `${w - cw}px`;
      frame2.style.height = `${ch}px`;
    }
    if (frame3) {
      frame3.style.left   = '0px';
      frame3.style.top    = `${ch}px`;
      frame3.style.width  = `${w}px`;
      frame3.style.height = `${h - ch}px`;
    }

    if (label1) {
      label1.style.left   = '';
      label1.style.top    = '';
      label1.style.right  = `${w - cw}px`;
      label1.style.bottom = `${h - ch}px`;
    }
    if (label2) {
      label2.style.right  = '';
      label2.style.top    = '';
      label2.style.left   = `${cw}px`;
      label2.style.bottom = `${h - ch}px`;
    }
    if (label3) {
      label3.style.left   = '0px';
      label3.style.right  = '0px';
      label3.style.bottom = '';
      label3.style.top    = `${ch}px`;
    }
  }

  function waitForMetadata(video, timeoutMs = 7000) {
    if (!video) return Promise.resolve();
    if (video.readyState >= 1 && isFinite(video.duration) && video.duration > 0) return Promise.resolve();

    return new Promise((resolve) => {
      let done = false;
      const cleanup = () => {
        if (done) return;
        done = true;
        video.removeEventListener('loadedmetadata', onReady);
        video.removeEventListener('canplay', onReady);
        video.removeEventListener('error', onReady);
      };
      const onReady = () => { cleanup(); resolve(); };
      video.addEventListener('loadedmetadata', onReady, { once: true });
      video.addEventListener('canplay',         onReady, { once: true });
      video.addEventListener('error',           onReady, { once: true });
      window.setTimeout(onReady, timeoutMs);
    });
  }

  function waitForCanPlay(video, timeoutMs = 7000) {
    if (!video) return Promise.resolve();
    if (video.readyState >= 3) return Promise.resolve();

    return new Promise((resolve) => {
      let done = false;
      const cleanup = () => {
        if (done) return;
        done = true;
        video.removeEventListener('canplay', onReady);
        video.removeEventListener('error',   onReady);
      };
      const onReady = () => { cleanup(); resolve(); };
      video.addEventListener('canplay', onReady, { once: true });
      video.addEventListener('error',   onReady, { once: true });
      window.setTimeout(onReady, timeoutMs);
    });
  }

  // Seeks a video and waits for the `seeked` event before resolving.
  // This prevents play() from firing while the seek is still pending.
  function seekToTime(video, targetTime, timeoutMs = 4000) {
    if (!video) return Promise.resolve();
    return new Promise((resolve) => {
      if (!Number.isFinite(video.duration) || video.duration <= 0) {
        resolve();
        return;
      }
      const t = Math.min(targetTime, Math.max(0, video.duration - 0.05));
      if (Math.abs(video.currentTime - t) < 0.01) {
        resolve();
        return;
      }
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        video.removeEventListener('seeked', finish);
        video.removeEventListener('error',  finish);
        resolve();
      };
      video.addEventListener('seeked', finish, { once: true });
      video.addEventListener('error',  finish, { once: true });
      window.setTimeout(finish, timeoutMs);
      try { video.currentTime = t; } catch { finish(); }
    });
  }

  function raf() {
    return new Promise((r) => requestAnimationFrame(r));
  }

  function safePause(video) {
    try { if (video && !video.paused) video.pause(); } catch { /* ignore */ }
  }

  function safePlay(video) {
    try { if (video && video.paused) video.play().catch(() => {}); } catch { /* ignore */ }
  }

  function getDuration(video) {
    if (!video) return NaN;
    const d = video.duration;
    return Number.isFinite(d) && d > 0 ? d : NaN;
  }

  function getProgress01(video) {
    const duration = getDuration(video);
    if (!Number.isFinite(duration)) return NaN;
    return clamp01(video.currentTime / duration);
  }

  function progressToTime(video, progress01) {
    const duration = getDuration(video);
    if (!Number.isFinite(duration)) return NaN;
    // Avoid seeking exactly to duration to prevent ended/loop boundary quirks.
    return Math.min(duration - 0.05, Math.max(0, duration * clamp01(progress01)));
  }

  class VideoTripletController {
    constructor(tripletEl, sceneIndex, sceneCount) {
      this.tripletEl  = tripletEl;
      this.sceneIndex = sceneIndex;
      this.sceneCount = sceneCount;
      this.isActive   = false;
      this.isUserPaused = false;
      this.isPrimed   = false;
      this.primedProgress = 0; // set after prime() resolves

      this.videos = {
        ours:     getRoleVideo(tripletEl, 'ours'),
        baseline: getRoleVideo(tripletEl, 'baseline'),
        gt:       getRoleVideo(tripletEl, 'gt'),
      };

      this.reference = this.videos.ours || this.videos.baseline || this.videos.gt;
      this.__all = [this.videos.ours, this.videos.baseline, this.videos.gt].filter(Boolean);

      this.__frame1  = tripletEl.querySelector('.twentytwenty-frame-1');
      this.__frame2  = tripletEl.querySelector('.twentytwenty-frame-2');
      this.__frame3  = tripletEl.querySelector('.twentytwenty-frame-3');
      this.__overlay = tripletEl.querySelector('.twentytwenty-overlay');
      this.__label1  = tripletEl.querySelector('.twentytwenty-label-1');
      this.__label2  = tripletEl.querySelector('.twentytwenty-label-2');
      this.__label3  = tripletEl.querySelector('.twentytwenty-label-3');

      // Cache on element for faster clip updates.
      tripletEl.__qualOurs     = this.videos.ours;
      tripletEl.__qualBaseline = this.videos.baseline;
      tripletEl.__qualGt       = this.videos.gt;
      tripletEl.__frame1       = this.__frame1;
      tripletEl.__frame2       = this.__frame2;
      tripletEl.__frame3       = this.__frame3;
      tripletEl.__overlay      = this.__overlay;
      tripletEl.__label1       = this.__label1;
      tripletEl.__label2       = this.__label2;
      tripletEl.__label3       = this.__label3;

      setSplitClip(tripletEl, SPLIT_DEFAULT_X, SPLIT_DEFAULT_Y);

      this._lastSyncAt    = 0;
      this._isTimeSyncing = false;
      this._activationSeq = 0;
      this._syncSeekDepth = 0;
      this._syncLoopRaf = 0;
      this._primePromise  = this._prime();

      this._bindSyncEvents();
      this._ensurePlaybackButton();
      this._ensurePlaybackFeedback();
      this._bindTripletTapToggle();
      this._updatePlaybackButtonState();
    }

    _ensurePlaybackFeedback() {
      let root = this.tripletEl.querySelector('.qualitative-video-playback-feedback');
      if (!root) {
        root = document.createElement('div');
        root.className = 'qualitative-video-playback-feedback';
        root.setAttribute('aria-hidden', 'true');
        root.innerHTML = `
          <div class="qualitative-video-playback-feedback__backdrop">
            <svg class="qualitative-video-playback-feedback__icon qualitative-video-playback-feedback__icon--play" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path fill="currentColor" d="M8 5v14l11-7z"/>
            </svg>
            <svg class="qualitative-video-playback-feedback__icon qualitative-video-playback-feedback__icon--pause" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
            </svg>
          </div>`;
        this.tripletEl.appendChild(root);
      }
      this.feedbackEl = root;
    }

    _clearPlaybackFeedback() {
      if (this._feedbackTimeoutId) {
        clearTimeout(this._feedbackTimeoutId);
        this._feedbackTimeoutId = 0;
      }
      if (this.feedbackEl) {
        this.feedbackEl.classList.remove('is-visible', 'is-pause', 'is-play');
      }
    }

    /**
     * @param {'play'|'pause'} mode — icon to show (matches state *after* the user action).
     */
    _flashPlaybackFeedback(mode) {
      this._ensurePlaybackFeedback();
      const el = this.feedbackEl;
      if (!el) return;
      this._clearPlaybackFeedback();
      void el.offsetWidth;
      el.classList.add('is-visible', mode === 'pause' ? 'is-pause' : 'is-play');
      this._feedbackTimeoutId = window.setTimeout(() => {
        el.classList.remove('is-visible', 'is-pause', 'is-play');
        this._feedbackTimeoutId = 0;
      }, FEEDBACK_FLASH_MS);
    }

    _ensurePlaybackButton() {
      let button = this.tripletEl.querySelector('.qualitative-video-playback-toggle');
      if (!button) {
        button = document.createElement('button');
        button.type = 'button';
        button.className = 'qualitative-video-playback-toggle';
        button.innerHTML = '<span class="qualitative-video-playback-toggle__text"></span>';
        this.tripletEl.appendChild(button);
      }
      button.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        this.togglePlayback();
      });
      this.playbackButton = button;
      this.playbackButtonText = button.querySelector('.qualitative-video-playback-toggle__text');
    }

    _updatePlaybackButtonState() {
      if (!this.playbackButton) return;
      const label = this.isUserPaused ? 'Play' : 'Pause';
      this.playbackButton.setAttribute('aria-pressed', this.isUserPaused ? 'true' : 'false');
      this.playbackButton.setAttribute('aria-label', `${label} comparison videos`);
      if (this.playbackButtonText) this.playbackButtonText.textContent = label;
    }

    setUserPaused(shouldPause, options = {}) {
      const { feedback = false } = options;
      this.isUserPaused = !!shouldPause;
      this._updatePlaybackButtonState();

      if (!this.isActive) return;
      if (this.isUserPaused) {
        this.__all.forEach((v) => safePause(v));
      } else {
        this.__all.forEach((v) => safePlay(v));
      }
      if (feedback) {
        this._flashPlaybackFeedback(this.isUserPaused ? 'pause' : 'play');
      }
    }

    togglePlayback() {
      this.setUserPaused(!this.isUserPaused, { feedback: true });
    }

    /**
     * Toggle play/pause when the user taps/clicks the comparison area (not only the small control).
     * Uses a movement threshold so dragging to adjust the split does not toggle.
     */
    _bindTripletTapToggle() {
      const el = this.tripletEl;
      let tapDown = null;

      el.addEventListener('pointerdown', (e) => {
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        if (e.target.closest('.qualitative-video-playback-toggle')) return;
        tapDown = { x: e.clientX, y: e.clientY };
      }, { passive: true });

      el.addEventListener('pointerup', (e) => {
        if (!tapDown) return;
        const dx = Math.abs(e.clientX - tapDown.x);
        const dy = Math.abs(e.clientY - tapDown.y);
        if (dx > TAP_TOGGLE_MOVE_THRESHOLD_PX || dy > TAP_TOGGLE_MOVE_THRESHOLD_PX) tapDown = null;
      }, { passive: true });

      el.addEventListener('click', (e) => {
        if (!this.isActive) return;
        if (e.target.closest('.qualitative-video-playback-toggle')) return;
        if (!tapDown) return;
        const dx = Math.abs(e.clientX - tapDown.x);
        const dy = Math.abs(e.clientY - tapDown.y);
        tapDown = null;
        if (dx > TAP_TOGGLE_MOVE_THRESHOLD_PX || dy > TAP_TOGGLE_MOVE_THRESHOLD_PX) return;
        e.preventDefault();
        this.togglePlayback();
      });

      el.addEventListener('pointercancel', () => { tapDown = null; }, { passive: true });
    }

    async _prime() {
      if (!this.reference) return;

      setSplitClip(this.tripletEl, SPLIT_DEFAULT_X, SPLIT_DEFAULT_Y);

      await Promise.all([
        waitForMetadata(this.videos.ours),
        waitForMetadata(this.videos.baseline),
        waitForMetadata(this.videos.gt),
      ]);

      if (this.reference.videoWidth && this.reference.videoHeight) {
        this.tripletEl.style.aspectRatio =
          `${this.reference.videoWidth} / ${this.reference.videoHeight}`;
      }

      const fraction = clamp01((this.sceneIndex + 0.5) / this.sceneCount);
      this.primedProgress = fraction;

      // Seek every video to the same normalized timeline position.
      await Promise.all(this.__all.map((v) => {
        const t = progressToTime(v, fraction);
        if (!Number.isFinite(t)) return Promise.resolve();
        return seekToTime(v, t);
      }));
      this.__all.forEach((v) => safePause(v));
      this.isPrimed = true;
    }

    whenPrimed() {
      return this._primePromise;
    }

    // Only the reference video drives time-sync; others follow.
    // This prevents the three videos from fighting each other.
    _bindSyncEvents() {
      const reference = this.reference;
      if (!reference) return;

      const others = this.__all.filter((v) => v !== reference);

      // Keep play/pause/buffer in lock-step across all three videos.
      this.__all.forEach((source) => {
        source.addEventListener('play', () => {
          if (!this.isActive) { safePause(source); return; }
          if (this.isUserPaused) { safePause(source); return; }
          this.__all.forEach((t) => { if (t && t !== source) safePlay(t); });
        });
        source.addEventListener('pause', () => {
          if (!this.isActive) return;
          if (this.isUserPaused) return;
          if (this._syncSeekDepth > 0) return;
          this.__all.forEach((t) => { if (t && t !== source) safePause(t); });
        });
        // If one video stalls/buffers (not a seek), pause the others so they stay in sync.
        // Ignore waiting events caused by programmatic seeks (source.seeking === true)
        // because those are triggered by our own sync corrections and should not
        // cascade into pausing the reference video.
        source.addEventListener('waiting', () => {
          if (!this.isActive || this.isUserPaused) return;
          if (source.seeking) return;
          this.__all.forEach((t) => { if (t && t !== source) safePause(t); });
        });
        // When the stalled video resumes, re-align all non-reference videos to the
        // reference position before resuming so a buffer stall doesn't leave residual drift.
        source.addEventListener('playing', () => {
          if (!this.isActive || this.isUserPaused) return;
          this._alignToReference();
          this.__all.forEach((t) => { if (t && t !== source) safePlay(t); });
          this._startSyncLoop();
        });
        source.addEventListener('ended', () => {
          if (!this.isActive || this.isUserPaused) return;
          this.__all.forEach((t) => { if (t && t !== source) safePause(t); });
          this._alignToReference(true);
          this.__all.forEach((t) => { if (t && t !== source) safePlay(t); });
          this._startSyncLoop();
        });
      });

      // Only reference drives time correction.
      const onTimeSync = () => {
        if (!this.isActive || this._isTimeSyncing) return;
        const now = performance.now();
        if (now - this._lastSyncAt < SYNC_THROTTLE_MS) return;
        this._lastSyncAt = now;

        const srcProgress = getProgress01(reference);
        if (!Number.isFinite(srcProgress)) return;

        this._isTimeSyncing = true;
        this._syncSeekDepth += 1;
        others.forEach((target) => {
          if (!target) return;
          const targetTime = progressToTime(target, srcProgress);
          if (!Number.isFinite(targetTime)) return;
          const absDrift = Math.abs(targetTime - target.currentTime);
          if (absDrift > SYNC_HARD_SEEK_THRESHOLD_SECONDS) {
            try { target.currentTime = targetTime; } catch { /* ignore */ }
          }
        });
        window.setTimeout(() => {
          this._syncSeekDepth = Math.max(0, this._syncSeekDepth - 1);
          this._isTimeSyncing = false;
        }, 120);
      };

      reference.addEventListener('timeupdate', onTimeSync);
      reference.addEventListener('seeking',    onTimeSync);
    }

    _alignToReference(force = false) {
      const refProgress = getProgress01(this.reference);
      if (!Number.isFinite(refProgress)) return;
      this.__all.forEach((v) => {
        if (!v || v === this.reference) return;
        const expectedTime = progressToTime(v, refProgress);
        if (!Number.isFinite(expectedTime)) return;
        const drift = Math.abs(v.currentTime - expectedTime);
        if (force || drift > 0.01) {
          try { v.currentTime = expectedTime; } catch { /* ignore */ }
        }
      });
    }

    _stopSyncLoop() {
      if (!this._syncLoopRaf) return;
      cancelAnimationFrame(this._syncLoopRaf);
      this._syncLoopRaf = 0;
    }

    _startSyncLoop() {
      if (this._syncLoopRaf) return;
      const tick = () => {
        this._syncLoopRaf = 0;
        if (!this.isActive || this.isUserPaused || !this.reference) return;

        if (this.reference.paused) {
          this.__all.forEach((v) => { if (v && v !== this.reference) safePause(v); });
          this._alignToReference(true);
          return;
        }

        const refProgress = getProgress01(this.reference);
        if (Number.isFinite(refProgress)) {
          this.__all.forEach((v) => {
            if (!v || v === this.reference) return;
            if (v.paused) safePlay(v);
            const expectedTime = progressToTime(v, refProgress);
            if (!Number.isFinite(expectedTime)) return;
            if (Math.abs(v.currentTime - expectedTime) > SYNC_HARD_SEEK_THRESHOLD_SECONDS) {
              try { v.currentTime = expectedTime; } catch { /* ignore */ }
            }
          });
        }

        this._syncLoopRaf = requestAnimationFrame(tick);
      };
      this._syncLoopRaf = requestAnimationFrame(tick);
    }

    setActive(shouldBeActive) {
      this.isActive = !!shouldBeActive;
      if (!this.isActive) {
        this._stopSyncLoop();
        this.__all.forEach((v) => safePause(v));
      }
    }

    async activate() {
      const activationSeq = ++this._activationSeq;

      // Wait for initial priming (metadata + seek) to complete.
      await this.whenPrimed();
      if (activationSeq !== this._activationSeq) return;

      // Pause all and re-seek every video back to the primed position.
      // We wait for each seeked event before proceeding so that play()
      // is never called while a seek is still in flight (which would
      // produce a black frame or cause one video to lag behind).
      this.__all.forEach((v) => safePause(v));
      await Promise.all(this.__all.map((v) => {
        const t = progressToTime(v, this.primedProgress);
        if (!Number.isFinite(t)) return Promise.resolve();
        return seekToTime(v, t);
      }));
      if (activationSeq !== this._activationSeq) return;

      this.isActive = true;

      // Give the browser one animation frame to reflow the newly visible
      // panel before computing clip dimensions.  Without this, clientWidth
      // / clientHeight can still be 0 right after removing is-hidden.
      await raf();
      if (activationSeq !== this._activationSeq) return;

      setSplitClip(this.tripletEl, SPLIT_DEFAULT_X, SPLIT_DEFAULT_Y);

      // Wait until every video has enough data to start rendering.
      await Promise.all([
        waitForCanPlay(this.videos.ours),
        waitForCanPlay(this.videos.baseline),
        waitForCanPlay(this.videos.gt),
      ]);
      if (activationSeq !== this._activationSeq) return;

      // One more alignment pass in case canplay caused a tiny currentTime drift.
      const refTime = this.reference?.currentTime;
      if (Number.isFinite(refTime)) {
        this._alignToReference();
      }

      if (this.isUserPaused) {
        this.__all.forEach((v) => safePause(v));
        this._updatePlaybackButtonState();
        return;
      }

      // Start all three videos together.
      await Promise.allSettled(
        this.__all.map((v) => {
          if (!v) return Promise.resolve();
          try { return v.play().catch(() => {}); } catch { return Promise.resolve(); }
        }),
      );
      this._startSyncLoop();
      this._updatePlaybackButtonState();
    }

    deactivate() {
      this._activationSeq++;
      this._stopSyncLoop();
      this._clearPlaybackFeedback();
      if (typeof this.tripletEl.__qualCancelSplitReset === 'function') {
        this.tripletEl.__qualCancelSplitReset();
      }
      this.tripletEl.__qualLastSplit = { sx: SPLIT_DEFAULT_X, sy: SPLIT_DEFAULT_Y };
      setSplitClip(this.tripletEl, SPLIT_DEFAULT_X, SPLIT_DEFAULT_Y);
      // Next time this panel is shown, autoplay unless the user pauses again while visible.
      this.isUserPaused = false;
      this.setActive(false);
      this._updatePlaybackButtonState();
    }

    bindHoverClip() {
      const el = this.tripletEl;
      let raf = 0;
      let pending = null;
      let bounds = null;
      let resetRaf = 0;

      const refreshBounds = () => { bounds = el.getBoundingClientRect(); };

      const cancelResetAnim = () => {
        if (resetRaf) {
          cancelAnimationFrame(resetRaf);
          resetRaf = 0;
        }
      };

      el.__qualCancelSplitReset = cancelResetAnim;

      const easeOutCubic = (t) => 1 - (1 - t) ** 3;

      const getLastSplit = () => {
        const last = el.__qualLastSplit;
        if (last && Number.isFinite(last.sx) && Number.isFinite(last.sy)) {
          return { sx: last.sx, sy: last.sy };
        }
        return { sx: SPLIT_DEFAULT_X, sy: SPLIT_DEFAULT_Y };
      };

      const runResetToDefault = () => {
        cancelResetAnim();
        const from = getLastSplit();
        const dx = SPLIT_DEFAULT_X - from.sx;
        const dy = SPLIT_DEFAULT_Y - from.sy;
        if (Math.hypot(dx, dy) < 0.002) {
          setSplitClip(el, SPLIT_DEFAULT_X, SPLIT_DEFAULT_Y);
          el.__qualLastSplit = { sx: SPLIT_DEFAULT_X, sy: SPLIT_DEFAULT_Y };
          return;
        }
        if (prefersReducedMotion()) {
          setSplitClip(el, SPLIT_DEFAULT_X, SPLIT_DEFAULT_Y);
          el.__qualLastSplit = { sx: SPLIT_DEFAULT_X, sy: SPLIT_DEFAULT_Y };
          return;
        }
        const start = performance.now();
        const tick = (now) => {
          const t = Math.min(1, (now - start) / SPLIT_RESET_MS);
          const e = easeOutCubic(t);
          const sx = from.sx + dx * e;
          const sy = from.sy + dy * e;
          setSplitClip(el, sx, sy);
          el.__qualLastSplit = { sx, sy };
          if (t < 1) {
            resetRaf = requestAnimationFrame(tick);
          } else {
            resetRaf = 0;
            el.__qualLastSplit = { sx: SPLIT_DEFAULT_X, sy: SPLIT_DEFAULT_Y };
          }
        };
        resetRaf = requestAnimationFrame(tick);
      };

      const apply = () => {
        raf = 0;
        if (!pending) return;
        const { clientX, clientY } = pending;
        pending = null;
        if (!bounds || bounds.width <= 0 || bounds.height <= 0) refreshBounds();
        const rect = bounds;
        if (!rect || rect.width <= 0 || rect.height <= 0) return;
        const sx = clamp01((clientX - rect.left) / rect.width);
        const sy = clamp01((clientY - rect.top) / rect.height);
        setSplitClip(el, sx, sy);
        el.__qualLastSplit = { sx, sy };
      };

      const onMove = (e) => {
        if (!e) return;
        cancelResetAnim();
        const pt = e.touches && e.touches.length ? e.touches[0] : e;
        pending = { clientX: pt.clientX, clientY: pt.clientY };
        if (raf) return;
        raf = window.requestAnimationFrame(apply);
      };

      el.addEventListener('pointerenter', () => {
        cancelResetAnim();
        refreshBounds();
        setSplitClip(el, SPLIT_DEFAULT_X, SPLIT_DEFAULT_Y);
        el.__qualLastSplit = { sx: SPLIT_DEFAULT_X, sy: SPLIT_DEFAULT_Y };
      });
      el.addEventListener('pointermove',  onMove, { passive: true });
      el.addEventListener('pointerleave', () => {
        bounds = null;
        runResetToDefault();
      });
      el.addEventListener('touchstart', (e) => {
        refreshBounds();
        cancelResetAnim();
        onMove(e);
      }, { passive: true });
      el.addEventListener('touchmove',  onMove, { passive: true });
      el.addEventListener('touchend',   () => {
        bounds = null;
        runResetToDefault();
      }, { passive: true });

      window.addEventListener('resize', refreshBounds, { passive: true });
    }
  }

  function activatePanel(switcher, activeModel) {
    const panels = switcher.querySelectorAll('.qualitative-model-panel');
    const tabs   = switcher.querySelectorAll('.qualitative-model-tab');

    tabs.forEach((t) => {
      const isActive = t.dataset.model === activeModel;
      t.classList.toggle('is-active', isActive);
      t.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    panels.forEach((panel) => {
      const isVisible = panel.dataset.modelPanel === activeModel;
      panel.classList.toggle('is-hidden', !isVisible);

      const controllers = panel.__videoTripletControllers || [];
      controllers.forEach((c) => {
        if (isVisible) {
          void c.activate();
        } else {
          c.deactivate();
        }
      });
    });
  }

  function initVideoComparisonBlock(switcher) {
    const sceneCount = parseSceneCount(switcher);
    const panels = switcher.querySelectorAll('.qualitative-model-panel');

    panels.forEach((panel) => {
      const triplets = panel.querySelectorAll('[data-video-triplet]');
      const controllers = [];
      triplets.forEach((tripletEl) => {
        const idx = Number(tripletEl.dataset.sceneIndex);
        const sceneIndex = Number.isFinite(idx) ? idx : 0;
        const controller = new VideoTripletController(tripletEl, sceneIndex, sceneCount);
        controller.bindHoverClip();
        controllers.push(controller);
      });
      panel.__videoTripletControllers = controllers;
    });

    const tabs = Array.from(switcher.querySelectorAll('.qualitative-model-tab'));
    if (!tabs.length) {
      // No tab switcher — activate all panels directly.
      panels.forEach((panel) => {
        (panel.__videoTripletControllers || []).forEach((c) => void c.activate());
      });
      return;
    }
    const activeTab = tabs.find((t) => t.classList.contains('is-active')) || tabs[0];
    activatePanel(switcher, activeTab.dataset.model);

    tabs.forEach((tab) => {
      tab.addEventListener('click', () => activatePanel(switcher, tab.dataset.model));
    });
  }

  function init() {
    const switchers = document.querySelectorAll('[data-qualitative-video-switcher]');
    if (!switchers.length) return;
    switchers.forEach((switcher) => initVideoComparisonBlock(switcher));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
