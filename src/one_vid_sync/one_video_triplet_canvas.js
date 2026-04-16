/**
 * Single-video timeline per merged hstack triplet cell; three panes drawn via Canvas2D.
 * Eliminates multi-HTMLVideoElement clock drift.
 */
(() => {
  if (window.__SITE_LITE__) return;

  const DEFAULT_SCENE_COUNT = 5;
  const SPLIT_DEFAULT_X = 0.5;
  const SPLIT_DEFAULT_Y = 0.5;
  const TAP_THRESHOLD_PX = 12;
  const FEEDBACK_FLASH_MS = 850;

  function clamp01(x) {
    return Math.min(1, Math.max(0, Number.isFinite(x) ? x : 0.5));
  }

  class MergedTripletCanvasCell {
    /**
     * @param {HTMLElement} cell - `[data-merged-triplet]`
     */
    constructor(cell) {
      this.cell = cell;
      this.video = cell.querySelector('.merged-triplet-source');
      this.canvas = cell.querySelector('.merged-triplet-canvas');
      if (!this.video || !this.canvas) return;

      this.ctx = this.canvas.getContext('2d', { alpha: false });
      this.overlay = cell.querySelector('.twentytwenty-overlay');
      this.frame1 = cell.querySelector('.twentytwenty-frame-1');
      this.frame2 = cell.querySelector('.twentytwenty-frame-2');
      this.frame3 = cell.querySelector('.twentytwenty-frame-3');
      this.label1 = cell.querySelector('.twentytwenty-label-1');
      this.label2 = cell.querySelector('.twentytwenty-label-2');
      this.label3 = cell.querySelector('.twentytwenty-label-3');

      this.sceneIndex = Number(cell.dataset.sceneIndex) || 0;
      this.sceneCount = Number(cell.dataset.sceneCount) || DEFAULT_SCENE_COUNT;

      this.sx = SPLIT_DEFAULT_X;
      this.sy = SPLIT_DEFAULT_Y;
      this.isUserPaused = false;
      this._raf = 0;
      this._feedbackTimeout = 0;
      this._cssW = 0;
      this._cssH = 0;

      this._onResize = () => {
        this.resizeCanvas();
        this.drawFrame();
      };
      this._resizeObserver = new ResizeObserver(this._onResize);
      this._resizeObserver.observe(cell);

      this._tick = () => {
        this._raf = 0;
        if (!this.video || this.video.paused) return;
        this.drawFrame();
        this._raf = requestAnimationFrame(this._tick);
      };

      this._ensurePlaybackButton();
      this._ensurePlaybackFeedback();
      this._bindPanelVisibility();
      this._bindPlaybackEvents();
      this._bindPointerSplit();

      this._onLoaded = () => this._seekToSceneAndMaybePlay();
      if (this.video.readyState >= 1 && Number.isFinite(this.video.duration) && this.video.duration > 0) {
        this._onLoaded();
      } else {
        this.video.addEventListener('loadedmetadata', this._onLoaded, { once: true });
        this.video.addEventListener('canplay', this._onLoaded, { once: true });
      }
    }

    _panelHidden() {
      const panel = this.cell.closest('.qualitative-model-panel');
      return !!(panel && panel.classList.contains('is-hidden'));
    }

    _bindPanelVisibility() {
      const panel = this.cell.closest('.qualitative-model-panel');
      if (!panel) return;
      const sync = () => {
        if (this._panelHidden()) {
          this._stopRaf();
          try {
            this.video.pause();
          } catch { /* ignore */ }
          this.drawFrame();
        } else if (!this.isUserPaused) {
          this.video.play().catch(() => {});
          this._startRafIfPlaying();
        }
      };
      const mo = new MutationObserver(sync);
      mo.observe(panel, { attributes: true, attributeFilter: ['class'] });
      sync();
    }

    resizeCanvas() {
      const cell = this.cell;
      const w = cell.clientWidth;
      const h = cell.clientHeight;
      if (w <= 0 || h <= 0) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
      this._cssW = w;
      this._cssH = h;
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.canvas.style.width = `${w}px`;
      this.canvas.style.height = `${h}px`;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    /**
     * Map merged hstack [Ours | Baseline | GT] into three display regions (same as legacy clip split).
     */
    drawFrame() {
      const v = this.video;
      const ctx = this.ctx;
      const w = this._cssW;
      const h = this._cssH;
      const vw = v.videoWidth;
      const vh = v.videoHeight;
      if (!w || !h || !vw || !vh) return;

      const sx = this.sx;
      const sy = this.sy;
      const cw = Math.round(sx * w);
      const ch = Math.round(sy * h);
      const thirdW = vw / 3;

      ctx.clearRect(0, 0, w, h);

      // Ours — top-left: column 0
      ctx.drawImage(
        v,
        0,
        0,
        (cw / w) * thirdW,
        (ch / h) * vh,
        0,
        0,
        cw,
        ch,
      );

      // Baseline — top-right: column 1
      ctx.drawImage(
        v,
        thirdW + (cw / w) * thirdW,
        0,
        ((w - cw) / w) * thirdW,
        (ch / h) * vh,
        cw,
        0,
        w - cw,
        ch,
      );

      // GT — bottom strip: column 2
      ctx.drawImage(
        v,
        2 * thirdW,
        (ch / h) * vh,
        thirdW,
        ((h - ch) / h) * vh,
        0,
        ch,
        w,
        h - ch,
      );

      this._applyOverlayLayout(w, h, cw, ch);
    }

    _applyOverlayLayout(w, h, cw, ch) {
      const overlay = this.overlay;
      if (overlay) {
        overlay.style.width = `${w}px`;
        overlay.style.height = `${h}px`;
      }
      const f1 = this.frame1;
      const f2 = this.frame2;
      const f3 = this.frame3;
      if (f1) {
        f1.style.left = '0px';
        f1.style.top = '0px';
        f1.style.width = `${cw}px`;
        f1.style.height = `${ch}px`;
      }
      if (f2) {
        f2.style.left = `${cw}px`;
        f2.style.top = '0px';
        f2.style.width = `${w - cw}px`;
        f2.style.height = `${ch}px`;
      }
      if (f3) {
        f3.style.left = '0px';
        f3.style.top = `${ch}px`;
        f3.style.width = `${w}px`;
        f3.style.height = `${h - ch}px`;
      }
      const l1 = this.label1;
      const l2 = this.label2;
      const l3 = this.label3;
      if (l1) {
        l1.style.left = '';
        l1.style.top = '';
        l1.style.right = `${w - cw}px`;
        l1.style.bottom = `${h - ch}px`;
      }
      if (l2) {
        l2.style.right = '';
        l2.style.top = '';
        l2.style.left = `${cw}px`;
        l2.style.bottom = `${h - ch}px`;
      }
      if (l3) {
        l3.style.left = '0px';
        l3.style.right = '0px';
        l3.style.bottom = '';
        l3.style.top = `${ch}px`;
      }
    }

    applySplit(sx, sy) {
      this.sx = clamp01(sx);
      this.sy = clamp01(sy);
      if (!this._cssW || !this._cssH) this.resizeCanvas();
      this.drawFrame();
    }

    _seekToSceneAndMaybePlay() {
      const v = this.video;
      if (!v.duration || !Number.isFinite(v.duration)) return;

      if (v.videoWidth && v.videoHeight) {
        this.cell.style.aspectRatio = `${v.videoWidth / 3} / ${v.videoHeight}`;
      }

      const t = ((this.sceneIndex + 0.5) / this.sceneCount) * v.duration;
      try {
        v.currentTime = t;
      } catch { /* ignore */ }

      this.resizeCanvas();
      this.applySplit(SPLIT_DEFAULT_X, SPLIT_DEFAULT_Y);

      if (this._panelHidden()) {
        try {
          v.pause();
        } catch { /* ignore */ }
        this.drawFrame();
        return;
      }

      if (!this.isUserPaused) {
        v.play()
          .then(() => {
            this._startRafIfPlaying();
          })
          .catch(() => {});
      } else {
        this.drawFrame();
      }
    }

    _startRafIfPlaying() {
      if (this._raf) return;
      if (!this.video || this.video.paused) return;
      this._raf = requestAnimationFrame(this._tick);
    }

    _stopRaf() {
      if (!this._raf) return;
      cancelAnimationFrame(this._raf);
      this._raf = 0;
    }

    _ensurePlaybackButton() {
      let button = this.cell.querySelector('.qualitative-video-playback-toggle');
      if (!button) {
        button = document.createElement('button');
        button.type = 'button';
        button.className = 'qualitative-video-playback-toggle';
        button.innerHTML = '<span class="qualitative-video-playback-toggle__text"></span>';
        this.cell.appendChild(button);
      }
      this.playbackButton = button;
      this.playbackButtonText = button.querySelector('.qualitative-video-playback-toggle__text');
    }

    _ensurePlaybackFeedback() {
      let root = this.cell.querySelector('.qualitative-video-playback-feedback');
      if (!root) {
        root = document.createElement('div');
        root.className = 'qualitative-video-playback-feedback';
        root.setAttribute('aria-hidden', 'true');
        root.innerHTML =
          '<div class="qualitative-video-playback-feedback__backdrop">' +
          '<svg class="qualitative-video-playback-feedback__icon qualitative-video-playback-feedback__icon--play" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M8 5v14l11-7z"/></svg>' +
          '<svg class="qualitative-video-playback-feedback__icon qualitative-video-playback-feedback__icon--pause" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>' +
          '</div>';
        this.cell.appendChild(root);
      }
      this.feedbackEl = root;
    }

    _flashFeedback(mode) {
      const el = this.feedbackEl;
      if (!el) return;
      if (this._feedbackTimeout) {
        clearTimeout(this._feedbackTimeout);
        this._feedbackTimeout = 0;
      }
      el.classList.remove('is-visible', 'is-play', 'is-pause');
      void el.offsetWidth;
      el.classList.add('is-visible', mode === 'pause' ? 'is-pause' : 'is-play');
      this._feedbackTimeout = window.setTimeout(() => {
        el.classList.remove('is-visible', 'is-play', 'is-pause');
        this._feedbackTimeout = 0;
      }, FEEDBACK_FLASH_MS);
    }

    _updatePlaybackButtonState() {
      if (!this.playbackButton) return;
      const paused = this.isUserPaused;
      const label = paused ? 'Play' : 'Pause';
      this.playbackButton.setAttribute('aria-pressed', paused ? 'true' : 'false');
      this.playbackButton.setAttribute('aria-label', `${label} comparison video`);
      if (this.playbackButtonText) this.playbackButtonText.textContent = label;
    }

    _bindPlaybackEvents() {
      const v = this.video;

      v.addEventListener('play', () => {
        this._startRafIfPlaying();
      });
      v.addEventListener('pause', () => {
        this._stopRaf();
        this.drawFrame();
      });
      v.addEventListener('seeked', () => this.drawFrame());

      this.playbackButton.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        this._togglePlayback();
      });

      this._updatePlaybackButtonState();
    }

    _togglePlayback() {
      const willPlay = this.isUserPaused;
      if (willPlay && this._panelHidden()) return;
      this.isUserPaused = !willPlay;
      this._updatePlaybackButtonState();
      if (willPlay) {
        this.video.play().catch(() => {});
        this._flashFeedback('play');
      } else {
        this.video.pause();
        this._flashFeedback('pause');
      }
    }

    _bindPointerSplit() {
      const cell = this.cell;
      let downX = 0;
      let downY = 0;
      let didMove = false;

      cell.addEventListener('pointerleave', () => {
        this.applySplit(SPLIT_DEFAULT_X, SPLIT_DEFAULT_Y);
      });

      cell.addEventListener(
        'pointermove',
        (e) => {
          if (e.target.closest && e.target.closest('.qualitative-video-playback-toggle')) return;
          const rect = cell.getBoundingClientRect();
          const sx = clamp01((e.clientX - rect.left) / rect.width);
          const sy = clamp01((e.clientY - rect.top) / rect.height);
          this.applySplit(sx, sy);
          const dx = e.clientX - downX;
          const dy = e.clientY - downY;
          if (Math.hypot(dx, dy) > TAP_THRESHOLD_PX) didMove = true;
        },
        { passive: true },
      );

      cell.addEventListener('pointerdown', (e) => {
        if (e.target.closest && e.target.closest('.qualitative-video-playback-toggle')) return;
        downX = e.clientX;
        downY = e.clientY;
        didMove = false;
      });

      cell.addEventListener('click', (e) => {
        if (e.target.closest && e.target.closest('.qualitative-video-playback-toggle')) return;
        if (didMove) return;
        e.preventDefault();
        this._togglePlayback();
      });
    }
  }

  function init() {
    document.querySelectorAll('[data-merged-triplet]').forEach((cell) => {
      if (cell.dataset.mergedTripletCanvas === '1') return;
      if (!cell.querySelector('.merged-triplet-source') || !cell.querySelector('.merged-triplet-canvas')) return;
      cell.dataset.mergedTripletCanvas = '1';
      cell.classList.add('merged-triplet-canvas-wrap');
      new MergedTripletCanvasCell(cell);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
