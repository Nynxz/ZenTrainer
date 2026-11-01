// The Text size setting zooms the page (CSS zoom), which leaves numbers in two spaces: a box from
// getBoundingClientRect, a pointer's clientX/Y and innerWidth/Height are screen pixels, while
// offsetWidth, scrollTop and the px a style is given are the page's own. Divide the first by this
// to get the second.

/** The CSS zoom `el` is drawn at; the page's by default. 1 in a browser that can't tell. */
export function zoomOf(el: Element | null | undefined = document.documentElement): number {
  return el?.currentCSSZoom || 1
}
