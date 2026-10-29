"""Retake the README screenshots and the banner's panels from a running ZenTrainer.

    uv run --with playwright --with pillow --no-project python docs/banner/shots.py \\
        --url http://127.0.0.1:8188 --run <run_id> --dataset <folder>

Point it at an install holding only demo content (the README's is an original character, Aurora,
made with MOSS and Qwen Image 2.1 and trained as a MOSS LoRA). The pages open in the Isotope theme
through the URL (?theme=isotope&mode=dark): the install's saved theme isn't touched, and writes to
its settings are blocked. Then render the banner: python docs/banner/render.py
"""

import argparse
import asyncio
import importlib.util
import io
from pathlib import Path
from urllib.parse import quote

from PIL import Image
from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parent / "images"
REPO = HERE.parents[1]


def theme_css(theme: str) -> str | None:
    """The theme as this checkout has it (the server may run another): its own module, no ComfyUI needed."""
    spec = importlib.util.spec_from_file_location("zt_themes", REPO / "engine" / "zentrainer" / "server" / "themes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.theme_css(theme, [REPO / "engine" / "themes"])
W, H = 1600, 1000


def save(png: bytes, name: str, width: int = W) -> None:
    img = Image.open(io.BytesIO(png))
    img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    out = IMAGES / f"{name}.webp"
    img.save(out, "WEBP", quality=88, method=6)
    print(out, img.size, f"{out.stat().st_size // 1024} KB")


async def main(url: str, run: str, dataset: str, chromium: str | None, mask: str | None, chart_run: str | None,
               subject: str, recipe: str, training_run: str | None) -> None:
    app = f"{url}/zentrainer"
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=chromium)
        page = await browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2, color_scheme="dark")

        async def no_writes(route):  # reads only, and SAM3 finding the mask (its result is a job file): the
            # demo install's settings, captions and masks are never written, whatever a page sends
            r = route.request
            if r.method == "GET" or (r.method == "POST" and r.url.split("?")[0].endswith("/segment")):
                await route.continue_()
            else:
                await route.abort()

        await page.route("**/*", no_writes)
        css = theme_css("isotope")
        if css:
            await page.route("**/zentrainer/api/themes/isotope/css", lambda r: r.fulfill(body=css, content_type="text/css"))
        # the dataset grid shows whole pictures, not cropped to square cards (a view preference kept in
        # this browser's local storage, so the install's own settings stay as they are)
        await page.add_init_script("localStorage.setItem('zentrainer.datasetView', JSON.stringify({ fit: 'contain' }))")

        async def go(path: str, wait: int = 2500) -> None:
            await page.goto(app + path + ("&" if "?" in path else "?") + "theme=isotope&mode=dark")
            await page.wait_for_load_state("networkidle")
            # no scrollbars in the pictures: the browser's, nor the app's own (ScrollArea's track)
            await page.add_style_tag(content="*{scrollbar-width:none!important}::-webkit-scrollbar{display:none!important}"
                                             ".scroll-area>.track{display:none!important}")
            await page.wait_for_timeout(wait)
            if mask:  # the demo install's own path, as a reader's would read
                await page.evaluate(
                    """m => { const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                    for (let n; (n = w.nextNode()); ) if (n.nodeValue.includes(m)) n.nodeValue = n.nodeValue.split(m).join('~/ComfyUI') }""",
                    mask,
                )

        # a run, and the banner's panels from it
        await go(f"/runs/{run}")
        await page.get_by_text("Speed", exact=True).first.click()  # the speed line on the chart (hidden by default)
        await page.wait_for_timeout(500)
        save(await page.screenshot(), "run")
        png = await page.locator(".layout .main section.card >> nth=1").screenshot()
        Image.open(io.BytesIO(png)).save(HERE / "el-samples.png")
        if chart_run:  # the chart can come from a longer run (a character's, with its likeness line)
            await go(f"/runs/{chart_run}")
        png = await page.locator(".layout .main section.card >> nth=0").screenshot()
        Image.open(io.BytesIO(png)).save(HERE / "el-loss.png")

        # New run, filled from the same run (Duplicate)
        await go(f"/runs/new?from={run}")
        save(await page.screenshot(), "new-run")

        # the README's composed pictures (datasets.html, simple.html) are made of these: a page without the
        # app's nav, or (whole) the full window, as a lightbox covers it
        async def panel(name: str, whole: bool = False) -> None:
            nav = await page.locator("aside, nav").first.bounding_box()
            x = 0 if whole else (nav["x"] + nav["width"]) if nav else 250
            png = await page.screenshot(clip={"x": x, "y": 0, "width": W - x, "height": H})
            Image.open(io.BytesIO(png)).save(HERE / f"el-{name}.webp", "WEBP", quality=92, method=6)  # photos: PNG is 3 MB

        # the dataset, then its viewer
        await go(f"/datasets/{dataset}")
        await panel("ds-grid")
        # the page without the app's nav: toolbar, the grid, and the Health pentagon beside it
        nav = await page.locator("aside, nav").first.bounding_box()
        x = (nav["x"] + nav["width"]) if nav else 250
        head = await page.locator("h1").first.bounding_box()  # from below the page's title: the banner tags the panel itself
        y = round(head["y"] + head["height"] + 8) if head else 20
        png = await page.screenshot(clip={"x": x + 8, "y": y, "width": W - x - 16, "height": 540})
        Image.open(io.BytesIO(png)).save(HERE / "el-dataset.png")
        await page.locator(".card-item img").first.click()
        await page.wait_for_timeout(1500)
        await page.locator('[title="Thumbnails"]').first.click()  # the whole dataset beside the picture
        await page.wait_for_timeout(1500)
        await panel("ds-viewer", whole=True)
        await page.keyboard.press("Escape")

        # the mask editor, SAM3 finding the subject in the first picture (shown, not saved)
        first = await page.evaluate(f"fetch('{app}/api/datasets/{dataset}').then(r => r.json()).then(d => d.items.find(i => i.kind !== 'video').path)")
        await go(f"/datasets/{dataset}/mask?path={quote(first)}")
        await page.get_by_placeholder("what to keep, e.g. person").fill(subject)
        await page.get_by_role("button", name="Find").click()
        await page.wait_for_timeout(20000)
        await panel("ds-mask")

        # Simple mode, as one picture (docs/banner/simple.html): the recipes, a recipe set up, a run training, a finished one
        await go("/simple")
        await panel("simple")
        # a recipe set up: named, on the demo dataset (one it has: nothing is uploaded)
        await go(f"/simple/{recipe}")
        await page.get_by_placeholder("e.g. Grandma Rose").fill(dataset.capitalize())
        await page.get_by_role("button", name="A dataset I have").click()
        await page.locator(f'.ds-pick button[title="{dataset}"]').click()
        await page.wait_for_timeout(3000)
        await panel("simple-setup")
        if training_run:  # one still training, as Simple mode shows it
            await go(f"/simple/run/{training_run}")
            await panel("simple-training")
        await go(f"/simple/run/{run}")
        await panel("simple-run")
        await browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--chromium", help="a Chromium binary, if Playwright's own isn't installed")
    ap.add_argument("--mask", help="the demo install's ComfyUI folder: shown as ~/ComfyUI")
    ap.add_argument("--chart-run", help="take the banner's loss chart from this run instead (only the chart: no samples)")
    ap.add_argument("--subject", default="person", help="what SAM3 finds in the mask editor shot")
    ap.add_argument("--recipe", default="moss-character", help="the Simple mode recipe shown set up on the dataset")
    ap.add_argument("--training-run", help="a run still training, for Simple mode's picture (else its last one is kept)")
    a = ap.parse_args()
    asyncio.run(main(a.url, a.run, a.dataset, a.chromium, a.mask, a.chart_run, a.subject, a.recipe, a.training_run))
