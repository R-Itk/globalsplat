(() => {
  function activatePanel(switcher, model) {
    const tabs = switcher.querySelectorAll(".qualitative-model-tab");
    tabs.forEach((t) => {
      const isActive = t.dataset.model === model;
      t.classList.toggle("is-active", isActive);
      t.setAttribute("aria-selected", isActive ? "true" : "false");
    });

    const panels = switcher.querySelectorAll(".qualitative-model-panel");
    panels.forEach((p) => {
      const shouldShow = p.dataset.modelPanel === model;
      p.classList.toggle("is-hidden", !shouldShow);
    });

    // Recompute geometry after visibility change.
    if (window.jQuery) {
      window.jQuery(window).trigger("resize.twentytwenty");
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const switchers = document.querySelectorAll("[data-qualitative-model-switcher]");
    switchers.forEach((switcher) => {
      const tabs = Array.from(switcher.querySelectorAll(".qualitative-model-tab"));
      if (tabs.length === 0) return;

      const activeTab = tabs.find((t) => t.classList.contains("is-active")) || tabs[0];
      const activeModel = activeTab.dataset.model;
      activatePanel(switcher, activeModel);

      tabs.forEach((tab) => {
        tab.addEventListener("click", () => activatePanel(switcher, tab.dataset.model));
      });
    });
  });
})();

