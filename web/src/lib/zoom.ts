// The Text size setting zooms the page (CSS zoom), which leaves numbers in two spaces: a box from
// getBoundingClientRect, a pointer's clientX/Y and innerWidth/Height are screen pixels, while
// offsetWidth, scrollTop and the px a style is given are the page's own. Divide the first by this
// to get the second.

/** The CSS zoom `el` is drawn at; the page's by default. */
export function zoomOf(el: Element | null | undefined = document.documentElement): number {
  if (!el) return 1
  if (typeof el.currentCSSZoom === 'number') return el.currentCSSZoom || 1
  // an engine that can't say: its box on screen against its own width
  const w = el instanceof HTMLElement ? el.offsetWidth : 0
  return (w && el.getBoundingClientRect().width / w) || 1
}
