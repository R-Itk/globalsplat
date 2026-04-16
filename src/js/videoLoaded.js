(function () {
  if (window.__SITE_LITE__) {
    window.addEventListener('DOMContentLoaded', function () {
      document.querySelectorAll('.method-cell').forEach(function (cell) {
        cell.classList.add('is-video-loaded');
      });
    });
    return;
  }

  function markCellLoaded(cell) {
    if (!cell) return;
    cell.classList.add('is-video-loaded');
  }

  function isComparisonCell(cell) {
    if (!cell) return false;
    return !!cell.querySelector('[data-comparison-root]');
  }

  function isVideoReady(video) {
    if (!video) return false;
    // HAVE_CURRENT_DATA (2) means we can render the first frame.
    return video.readyState >= 2;
  }

  function bindSingleVideoCell(cell, video) {
    if (!cell || !video) return;
    if (isVideoReady(video)) markCellLoaded(cell);
    video.addEventListener('loadeddata', function () { markCellLoaded(cell); }, { once: true });
    video.addEventListener('canplay', function () { markCellLoaded(cell); }, { once: true });
    video.addEventListener('playing', function () { markCellLoaded(cell); }, { once: true });
    video.addEventListener('error', function () { markCellLoaded(cell); }, { once: true });
  }

  function bindMultiVideoCell(cell, videos) {
    if (!cell || !videos || !videos.length) return;

    var settledCount = 0;
    var settled = new WeakSet();

    function markVideoSettled(video) {
      if (!video || settled.has(video)) return;
      settled.add(video);
      settledCount += 1;
      if (settledCount >= videos.length) {
        markCellLoaded(cell);
      }
    }

    videos.forEach(function (video) {
      if (isVideoReady(video)) markVideoSettled(video);
      video.addEventListener('loadeddata', function () { markVideoSettled(video); }, { once: true });
      video.addEventListener('canplay', function () { markVideoSettled(video); }, { once: true });
      video.addEventListener('playing', function () { markVideoSettled(video); }, { once: true });
      video.addEventListener('error', function () { markVideoSettled(video); }, { once: true });
    });
  }

  window.addEventListener('DOMContentLoaded', function () {
    var cells = document.querySelectorAll('.method-cell');
    cells.forEach(function (cell) {
      // Comparison sliders have dedicated loading logic that waits for BOTH synced videos.
      if (isComparisonCell(cell)) return;

      var videos = Array.from(cell.querySelectorAll('video'));
      if (!videos.length) return;
      if (videos.length === 1) {
        bindSingleVideoCell(cell, videos[0]);
      } else {
        bindMultiVideoCell(cell, videos);
      }
    });
  });
})();

