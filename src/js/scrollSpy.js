function initScrollSpy() {
  // Scroll spy for top nav: keep the active item stable until the next
  // nav section is reached.
  const targets = Array.from(document.querySelectorAll('.scroll-target'));
  const navLinks = Array.from(document.querySelectorAll('.topnav-link'));

  if (!targets.length || !navLinks.length) return;

  const sectionIds = new Set(
    navLinks
      .map((link) => link.getAttribute('href') || '')
      .map((href) => href.replace(/^#/, ''))
      .filter(Boolean)
  );

  const navTargets = targets.filter((el) => {
    const id = el.getAttribute('id');
    return id && sectionIds.has(id);
  });

  if (!navTargets.length) return;

  const sectionOrdered = navLinks
    .map((link) => link.getAttribute('href') || '')
    .map((href) => href.replace(/^#/, ''))
    .filter(Boolean);

  const elById = new Map(
    navTargets
      .map((el) => el.getAttribute('id'))
      .map((id) => [id, document.getElementById(id)])
      .filter(([id, el]) => id && el)
  );

  const fallbackId = navTargets[0].getAttribute('id') || sectionOrdered[0];

  let currentId = null;
  let currentIndex = -1;
  let lastScrollY = window.scrollY;

  function setActive(id) {
    if (!id || id === currentId) return;

    navLinks.forEach((link) => {
      link.classList.remove('active');
      link.removeAttribute('aria-current');
    });
    const activeLink = navLinks.find(
      (link) => (link.getAttribute('href') || '') === `#${id}`
    );
    if (activeLink) {
      activeLink.classList.add('active');
      activeLink.setAttribute('aria-current', 'page');
    }

    currentId = id;
    currentIndex = sectionOrdered.indexOf(id);
  }

  function updateActive() {
    const activationY = window.scrollY + window.innerHeight / 2;

    let bestIndex = -1;
    for (let i = 0; i < sectionOrdered.length; i++) {
      const id = sectionOrdered[i];
      const el = elById.get(id);
      if (!el) continue;
      const absTop = el.getBoundingClientRect().top + window.scrollY;
      if (absTop <= activationY) bestIndex = i;
      else break;
    }

    const scrollBottom = window.scrollY + window.innerHeight;
    const docHeight = document.documentElement.scrollHeight;
    const isAtBottom = scrollBottom >= docHeight - 8;
    if (isAtBottom) bestIndex = sectionOrdered.length - 1;

    const nextId = bestIndex >= 0 ? sectionOrdered[bestIndex] : fallbackId;

    const scrollingDown = window.scrollY >= lastScrollY;
    const shouldAdvance = bestIndex > currentIndex && scrollingDown;
    const shouldRetreat = bestIndex < currentIndex && !scrollingDown;

    if (currentId === null) {
      setActive(nextId);
    } else if (shouldAdvance || shouldRetreat) {
      setActive(nextId);
    }

    lastScrollY = window.scrollY;
  }

  window.addEventListener('scroll', updateActive, { passive: true });
  window.addEventListener('resize', updateActive);

  updateActive();
}
