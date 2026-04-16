function initScrollReveal() {
  const STAGGER_DELAY_MS = 60;

  // Elements that should stagger their direct children
  const STAGGER_SELECTORS = [
    '.qualitative-image-grid',
    '.results-table tbody',
    '.table-carousel',
  ];

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Apply stagger delays via CSS custom property
  function applyStaggerDelays(parent) {
    const children = Array.from(parent.children);
    children.forEach((child, i) => {
      child.style.transitionDelay = `${i * STAGGER_DELAY_MS}ms`;
    });
  }

  // Reset stagger delays after animation so hover transitions aren't delayed
  function clearStaggerDelays(parent) {
    Array.from(parent.children).forEach((child) => {
      child.style.transitionDelay = '';
    });
  }

  // Promote matching descendants to stagger containers
  function markStaggerChildren(revealEl) {
    STAGGER_SELECTORS.forEach((sel) => {
      revealEl.querySelectorAll(sel).forEach((container) => {
        container.classList.add('reveal-stagger');
        if (!reduced) applyStaggerDelays(container);
      });
    });
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        el.classList.add('is-visible');

        // Trigger stagger children
        el.querySelectorAll('.reveal-stagger').forEach((container) => {
          // Small tick so stagger fires after the parent starts fading in
          requestAnimationFrame(() => container.classList.add('is-visible'));
        });

        // Clean up delays once animation completes (~700ms for longest child)
        if (!reduced) {
          window.setTimeout(() => {
            el.querySelectorAll('.reveal-stagger').forEach(clearStaggerDelays);
          }, 700 + STAGGER_DELAY_MS * 20);
        }

        observer.unobserve(el);
      });
    },
    {
      threshold: 0.08,
      rootMargin: '0px 0px -48px 0px',
    }
  );

  // Observe all .reveal elements
  document.querySelectorAll('.reveal').forEach((el) => {
    markStaggerChildren(el);
    observer.observe(el);
  });
}
