/** Cap for in-place scale(); keep in sync with --image-zoom-max-scale in tokens.css */
const IMAGE_ZOOM_MAX_SCALE = 1.55;
const IMAGE_ZOOM_VIEWPORT_MARGIN_PX = 16;
/** Match --image-zoom-duration in tokens.css; used for transition fallbacks */
const IMAGE_ZOOM_TRANSITION_MS = 550;
const IMAGE_ZOOM_TRANSITION_SAFE_MS = IMAGE_ZOOM_TRANSITION_MS + 150;

function maxScaleFittingViewport(rect) {
  const m = IMAGE_ZOOM_VIEWPORT_MARGIN_PX;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const { left, top, width: w, height: h } = rect;
  if (w <= 0 || h <= 0) return 1;
  const cx = left + w / 2;
  const cy = top + h / 2;
  const sLeft = (2 * (cx - m)) / w;
  const sRight = (2 * (vw - m - cx)) / w;
  const sTop = (2 * (cy - m)) / h;
  const sBottom = (2 * (vh - m - cy)) / h;
  return Math.max(1, Math.min(sLeft, sRight, sTop, sBottom, IMAGE_ZOOM_MAX_SCALE));
}

function getInlineTableFragment(src) {
  if (!src || !document || !document.querySelector) return null;
  const template = document.querySelector(`template[data-inline-fragment="${CSS.escape(src)}"]`);
  if (!template) return null;
  return template.innerHTML;
}

function renderTableFragment(el, html) {
  if (!el || typeof html !== 'string') return false;
  el.innerHTML = html;
  return true;
}

function loadTableFragments() {
  initImageZoom();

  const containers = document.querySelectorAll('[data-table-fragment]');
  containers.forEach((el) => {
    const src = el.getAttribute('data-table-fragment');
    if (!src) return;

    const inlineHtml = getInlineTableFragment(src);
    if (typeof inlineHtml === 'string' && inlineHtml.trim()) {
      renderTableFragment(el, inlineHtml);
      return;
    }

    fetch(src)
      .then((res) => (res.ok ? res.text() : Promise.reject(new Error(res.statusText))))
      .then((html) => {
        renderTableFragment(el, html);
      })
      .catch((err) => {
        // eslint-disable-next-line no-console
        console.error('Failed to load table fragment:', src, err);
      });
  });
}

let imageZoomInitialized = false;
let imageZoomActiveFigure = null;
let imageZoomActiveHost = null;
let imageZoomOpenGeneration = 0;

let imageZoomOverlay = null;

function initImageZoom() {
  if (imageZoomInitialized) return;
  imageZoomInitialized = true;

  const isZoomFigure = (figure) => Boolean(figure && figure.querySelector && figure.querySelector('img'));

  const getFigureFromTarget = (target) => {
    const figure = target && target.closest ? target.closest('[data-image-zoom]') : null;
    return isZoomFigure(figure) ? figure : null;
  };

  const ensureOverlay = () => {
    if (imageZoomOverlay) return imageZoomOverlay;

    imageZoomOverlay = document.querySelector('.table-zoom-overlay');
    if (imageZoomOverlay) return imageZoomOverlay;

    imageZoomOverlay = document.createElement('div');
    imageZoomOverlay.className = 'table-zoom-overlay';
    imageZoomOverlay.setAttribute('aria-hidden', 'true');
    imageZoomOverlay.style.opacity = '0';
    document.body.appendChild(imageZoomOverlay);
    return imageZoomOverlay;
  };

  const showOverlay = () => {
    const overlay = ensureOverlay();
    overlay.classList.add('is-visible');
  };

  const hideOverlay = () => {
    const overlay = imageZoomOverlay;
    if (!overlay) return;
    overlay.classList.remove('is-visible');
  };

  const resetFigureZoomVars = (figure) => {
    if (!figure) return;
    figure.style.removeProperty('--image-zoom-scale');
    figure.style.removeProperty('--image-zoom-slot-w');
    figure.style.removeProperty('--image-zoom-slot-h');
    figure.style.removeProperty('--image-zoom-img-w');
  };

  const getZoomHost = (figure) => {
    if (!figure || !figure.closest) return null;
    return figure.closest('.teaser-layout');
  };

  const setActiveZoomHost = (host) => {
    if (imageZoomActiveHost && imageZoomActiveHost !== host) {
      imageZoomActiveHost.classList.remove('is-image-zoom-host-active');
    }
    imageZoomActiveHost = host || null;
    if (imageZoomActiveHost) {
      imageZoomActiveHost.classList.add('is-image-zoom-host-active');
    }
  };

  const finishClose = (figure) => {
    figure.classList.remove('is-image-zoomed');
    figure.setAttribute('aria-expanded', 'false');
    resetFigureZoomVars(figure);
    setActiveZoomHost(null);
    imageZoomActiveFigure = null;
    hideOverlay();
  };

  const animateOpenFigure = (figure) => {
    if (!isZoomFigure(figure)) return;

    if (imageZoomActiveFigure && imageZoomActiveFigure !== figure) {
      imageZoomOpenGeneration += 1;
      const prev = imageZoomActiveFigure;
      prev.classList.remove('is-image-zoomed');
      prev.setAttribute('aria-expanded', 'false');
      resetFigureZoomVars(prev);
      setActiveZoomHost(null);
      imageZoomActiveFigure = null;
      hideOverlay();
    }

    const img = figure.querySelector('img');
    if (!img) return;

    const rect = figure.getBoundingClientRect();
    const scale = maxScaleFittingViewport(rect);
    const slotW = Math.round(rect.width);
    const slotH = Math.round(rect.height);
    const targetW = Math.max(1, Math.round(slotW * scale));
    const startW = Math.round(img.offsetWidth);

    figure.style.setProperty('--image-zoom-scale', scale.toFixed(4));
    figure.style.setProperty('--image-zoom-slot-w', `${slotW}px`);
    figure.style.setProperty('--image-zoom-slot-h', `${slotH}px`);
    figure.style.setProperty('--image-zoom-img-w', `${startW}px`);

    imageZoomActiveFigure = figure;
    setActiveZoomHost(getZoomHost(figure));
    figure.classList.add('is-image-zoomed');
    figure.setAttribute('aria-expanded', 'true');
    showOverlay();

    imageZoomOpenGeneration += 1;
    const gen = imageZoomOpenGeneration;
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        if (gen !== imageZoomOpenGeneration) return;
        if (imageZoomActiveFigure !== figure) return;
        figure.style.setProperty('--image-zoom-img-w', `${targetW}px`);
      });
    });
  };

  const animateCloseFigure = (figure) => {
    if (!figure || !imageZoomActiveFigure) return;
    if (imageZoomActiveFigure !== figure) return;

    imageZoomOpenGeneration += 1;

    const img = figure.querySelector('img');
    if (!img) {
      finishClose(figure);
      return;
    }

    const slotRaw = figure.style.getPropertyValue('--image-zoom-slot-w');
    const slotW = Math.round((slotRaw && parseFloat(slotRaw)) || figure.offsetWidth);
    const currentW = Math.round(img.getBoundingClientRect().width);
    if (Math.abs(currentW - slotW) <= 1) {
      finishClose(figure);
      return;
    }

    const onEnd = (ev) => {
      if (ev.target !== img || ev.propertyName !== 'width') return;
      window.clearTimeout(timer);
      img.removeEventListener('transitionend', onEnd);
      if (imageZoomActiveFigure === figure) finishClose(figure);
    };

    const timer = window.setTimeout(() => {
      img.removeEventListener('transitionend', onEnd);
      if (imageZoomActiveFigure === figure) finishClose(figure);
    }, IMAGE_ZOOM_TRANSITION_SAFE_MS);

    img.addEventListener('transitionend', onEnd);
    figure.style.setProperty('--image-zoom-img-w', `${slotW}px`);
  };

  const toggleFigure = (figure) => {
    if (!isZoomFigure(figure)) return;
    if (imageZoomActiveFigure === figure) animateCloseFigure(figure);
    else animateOpenFigure(figure);
  };

  document.addEventListener('click', (e) => {
    const target = e.target;

    // Don't start zoom when interacting with the info button itself.
    if (target && target.closest && target.closest('.hover-figure__btn')) return;
    if (target && target.closest && target.closest('.hover-figure__tooltip')) return;

    const figure = getFigureFromTarget(target);

    // If zoomed, clicking anywhere outside the zoom figure closes.
    if (imageZoomActiveFigure && !figure) {
      animateCloseFigure(imageZoomActiveFigure);
      return;
    }

    if (!figure) return;
    toggleFigure(figure);
  });

  // Optional escape-to-close for keyboard users.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!imageZoomActiveFigure) return;
    animateCloseFigure(imageZoomActiveFigure);
  });
}
