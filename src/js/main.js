document.addEventListener('DOMContentLoaded', () => {
  initScrollSpy();
  loadTableFragments();
  initExportPdf();
  if (!window.__SITE_LITE__) initComparisonSlider();
  initBibtexCopy();
  initScrollTopButton();
  initScrollReveal();
  initUltraCompactWidthSync();
});

function initBibtexCopy() {
  const button = document.getElementById('copy-bibtex-btn');
  const buttonText = button ? button.querySelector('.copy-bibtex-btn__text') : null;
  const bibtexCode = document.querySelector('#bibtex-code code');
  if (!button || !buttonText || !bibtexCode) return;

  const originalLabel = buttonText.textContent || 'Copy';

  const setCopiedState = () => {
    button.classList.add('is-copied');
    buttonText.textContent = 'Copied';
    window.setTimeout(() => {
      button.classList.remove('is-copied');
      buttonText.textContent = originalLabel;
    }, 450);
  };

  const fallbackCopy = (text) => {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.setAttribute('readonly', '');
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
  };

  button.addEventListener('click', async () => {
    const bibtexText = bibtexCode.textContent || '';
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(bibtexText);
      } else {
        fallbackCopy(bibtexText);
      }
      setCopiedState();
    } catch (error) {
      fallbackCopy(bibtexText);
      setCopiedState();
    }
  });
}

function initScrollTopButton() {
  const button = document.getElementById('scroll-top-btn');
  if (!button) return;

  const toggleVisibility = () => {
    if (window.scrollY > 320) button.classList.add('is-visible');
    else button.classList.remove('is-visible');
  };

  button.addEventListener('click', () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth',
    });
  });

  window.addEventListener('scroll', toggleVisibility, { passive: true });
  toggleVisibility();
}

function initUltraCompactWidthSync() {
  const targetWraps = document.querySelectorAll(
    '[data-video-container="ultra-compact-single-wrap"], [data-video-container="pointcloud-comparison"]',
  );
  if (!targetWraps.length) return;

  const referenceBlock = Array.from(document.querySelectorAll('.qualitative-video-comparison-block')).find((block) => {
    const heading = block.querySelector('.results-subgroup-title');
    return heading && heading.textContent.includes('12-VIEWS QUALITATIVE VIDEO COMPARISON');
  });
  if (!referenceBlock) return;

  const findReferenceCell = () => {
    return (
      referenceBlock.querySelector(
        '.qualitative-model-panel:not(.is-hidden) .qualitative-image-grid--compact .qualitative-image-cell',
      ) ||
      referenceBlock.querySelector('.qualitative-image-grid--compact .qualitative-image-cell')
    );
  };

  const applyReferenceWidth = () => {
    const cell = findReferenceCell();
    if (!cell) return false;
    const { width } = cell.getBoundingClientRect();
    if (!Number.isFinite(width) || width <= 0) return false;
    const pxWidth = `${Math.round(width)}px`;
    targetWraps.forEach((wrap) => {
      wrap.style.width = pxWidth;
      wrap.style.maxWidth = pxWidth;
    });
    return true;
  };

  let rafId = 0;
  let retryCount = 0;
  const MAX_RETRIES = 24;
  const scheduleApply = () => {
    if (rafId) return;
    rafId = window.requestAnimationFrame(() => {
      rafId = 0;
      const applied = applyReferenceWidth();
      if (!applied && retryCount < MAX_RETRIES) {
        retryCount += 1;
        window.setTimeout(scheduleApply, 50);
        return;
      }
      retryCount = 0;
    });
  };

  window.addEventListener('resize', scheduleApply, { passive: true });

  if (typeof ResizeObserver !== 'undefined') {
    const resizeObserver = new ResizeObserver(scheduleApply);
    const referenceCell = findReferenceCell();
    if (referenceCell) resizeObserver.observe(referenceCell);
  }

  if (typeof MutationObserver !== 'undefined') {
    const mutationObserver = new MutationObserver(scheduleApply);
    mutationObserver.observe(referenceBlock, {
      subtree: true,
      attributes: true,
      attributeFilter: ['class', 'style'],
    });
  }

  scheduleApply();
}
