(function() {
  function init() {
    if (!window.jQuery || !window.jQuery.fn || !window.jQuery.fn.twentytwenty) return;
    // The upstream twentytwenty plugin is designed around <img> elements.
    // Skip containers that contain no images (e.g., qualitative video tiles).
    window.jQuery(".twentytwenty-container").each(function () {
      if (!this || !this.querySelector || !this.querySelector("img")) return;
      window.jQuery(this).twentytwenty({
        default_offset_pct_x: 0.5,
        default_offset_pct_y: 0.5,
      });
    });
  }

  if (document.readyState === "complete") {
    init();
  } else {
    window.addEventListener("load", init);
  }
})();
