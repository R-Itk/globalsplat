function initTableCarousel() {
  const carousels = document.querySelectorAll('[data-table-carousel]')
  if (!carousels.length) return

  const wrapIndex = (index, len) => ((index % len) + len) % len

  carousels.forEach((carousel) => {
    const track = carousel.querySelector('.table-carousel__track')
    const viewport = carousel.querySelector('.table-carousel__viewport')
    const slides = track ? track.querySelectorAll('.table-carousel__slide') : []

    if (!track || !viewport || !slides || slides.length < 2) return

    const prevButton = carousel.querySelector('.table-carousel__nav--prev')
    const nextButton = carousel.querySelector('.table-carousel__nav--next')

    const len = slides.length
    let index = 0
    let lastViewportHeight = null

    // Build dot indicators and insert them below the carousel row.
    const dotsContainer = document.createElement('div')
    dotsContainer.className = 'table-carousel__dots'
    dotsContainer.setAttribute('role', 'tablist')
    dotsContainer.setAttribute('aria-label', 'Table navigation')
    carousel.insertAdjacentElement('afterend', dotsContainer)

    const dots = Array.from(slides).map((_, i) => {
      const dot = document.createElement('button')
      dot.className = 'table-carousel__dot' + (i === 0 ? ' is-active' : '')
      dot.setAttribute('aria-label', `Go to table ${i + 1} of ${len}`)
      dot.setAttribute('role', 'tab')
      dot.addEventListener('click', () => go(i))
      dotsContainer.appendChild(dot)
      return dot
    })

    const updateDots = () => {
      dots.forEach((dot, i) => dot.classList.toggle('is-active', i === index))
    }

    const updateSlideActive = () => {
      slides.forEach((s, i) => s.classList.toggle('is-active', i === index))
    }

    const update = () => {
      // Translate by the viewport width so each slide reliably snaps into view.
      const width = viewport.getBoundingClientRect().width || viewport.clientWidth || 0
      const snappedWidth = Math.round(width)
      const snappedOffset = Math.round(-index * snappedWidth)
      carousel.style.setProperty('--table-carousel-slide-width', `${snappedWidth}px`)
      track.style.transform = `translateX(${snappedOffset}px)`

      // Keep the nav buttons vertically centered relative to the active slide.
      // Transforms don't affect layout, so without this the viewport height may
      // reflect the taller slide, shifting centering for the smaller one.
      const activeSlide = slides[index]
      if (activeSlide && activeSlide.getBoundingClientRect) {
        const h = activeSlide.getBoundingClientRect().height
        if (h && h > 0) {
          if (lastViewportHeight === null || Math.abs(lastViewportHeight - h) > 1) {
            viewport.style.height = `${h}px`
            lastViewportHeight = h
          }
        }
      }
    }

    let resizeRaf = null
    const onResize = () => {
      if (resizeRaf) cancelAnimationFrame(resizeRaf)
      resizeRaf = requestAnimationFrame(update)
    }
    window.addEventListener('resize', onResize)

    // Fragments load asynchronously; observe for changes so we can recalc height.
    let mutationRaf = null
    const scheduleUpdateFromMutation = () => {
      if (mutationRaf) cancelAnimationFrame(mutationRaf)
      mutationRaf = requestAnimationFrame(() => {
        mutationRaf = null
        update()
      })
    }
    const observer = new MutationObserver(scheduleUpdateFromMutation)
    observer.observe(track, { childList: true, subtree: true })

    const go = (targetIndex) => {
      index = wrapIndex(targetIndex, len)
      update()
      updateDots()
      updateSlideActive()
    }

    const onPrev = () => go(index - 1)
    const onNext = () => go(index + 1)

    if (prevButton) prevButton.addEventListener('click', onPrev)
    if (nextButton) nextButton.addEventListener('click', onNext)

    carousel.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        onPrev()
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        onNext()
      }
    })

    // Set initial active state before first render.
    updateSlideActive()
    update()
  })
}

document.addEventListener('DOMContentLoaded', initTableCarousel)
