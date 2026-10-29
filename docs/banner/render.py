"""Render banner.html to docs/images/banner.webp (1600x800), how.html to docs/images/how-it-works.webp,
simple.html and datasets.html to docs/images/, badges.html to docs/images/badge-*.webp, and the Registry icon to
docs/images/icon.png (400x400: the banner's lotus mark on Isotope's black).

    uv run --with playwright --with pillow --no-project python docs/banner/render.py [--chromium PATH]

The panels are crops of the running app (isotope, dark); retake them when the UI changes.
"""

import argparse
import asyncio
import io
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "images" / "banner.webp"
ICON = HERE.parent / "images" / "icon.png"
HOW = HERE.parent / "images" / "how-it-works.webp"
SIMPLE = HERE.parent / "images" / "simple.webp"
DATASETS = HERE.parent / "images" / "datasets.webp"


async def main(chromium: str | None) -> None:
    async with async_playwright() as p:
        # file access: the composed pictures read their crops' colours (a canvas from local files)
        browser = await p.chromium.launch(executable_path=chromium, args=["--allow-file-access-from-files"])
        page = await browser.new_page(viewport={"width": 1600, "height": 800}, device_scale_factor=2)
        await page.goto((HERE / "banner.html").as_uri())
        await page.wait_for_load_state("networkidle")
        png = await page.locator(".banner").screenshot(omit_background=True)
        # How it works, the README's diagram (how.html), the same way
        await page.goto((HERE / "how.html").as_uri())
        await page.wait_for_load_state("networkidle")
        await page.evaluate("document.fonts.ready")
        how = await page.locator(".how").screenshot()
        # the composed pictures: Simple mode and Datasets, framed crops of the app
        sheets = {}
        for src, out in (("simple.html", SIMPLE), ("datasets.html", DATASETS)):
            await page.goto((HERE / src).as_uri())
            await page.wait_for_load_state("networkidle")
            await page.evaluate("document.fonts.ready.then(() => window.layout())")
            sheets[out] = await page.locator(".sheet").screenshot()
        # the badges under the banner, one picture each (each its own link in the README)
        await page.goto((HERE / "badges.html").as_uri())
        await page.wait_for_load_state("networkidle")
        await page.evaluate("document.fonts.ready")
        for b in await page.locator(".pad").all():
            out = HERE.parent / "images" / f"badge-{await b.get_attribute('id')}.webp"
            sheets[out] = await b.screenshot(omit_background=True)
        # the icon: the banner's own lotus, larger, on the banner's colours
        lotus = (HERE / "banner.html").read_text().split('<svg viewBox="-74 -88 148 102"', 1)[1].split("</svg>", 1)[0]
        await page.set_viewport_size({"width": 400, "height": 400})
        await page.set_content(
            '<body style="margin:0"><div id="i" style="width:400px;height:400px;display:grid;place-items:center;background:#0b0b0c">'
            f'<svg viewBox="-74 -88 148 102" width="300" height="207"{lotus}</svg></div></body>'
        )
        icon = await page.locator("#i").screenshot()
        await browser.close()
        Image.open(io.BytesIO(icon)).resize((400, 400), Image.LANCZOS).save(ICON, "PNG", optimize=True)
        print(ICON)
    # rendered at 2x for clean text, saved at 1x like the other READMEs' banners
    img = Image.open(io.BytesIO(png)).resize((1600, 800), Image.LANCZOS)
    img.save(OUT, "WEBP", quality=90, method=6)
    print(OUT, f"{OUT.stat().st_size // 1024} KB")
    img = Image.open(io.BytesIO(how))
    img.resize((img.width // 2, img.height // 2), Image.LANCZOS).save(HOW, "WEBP", quality=90, method=6)
    print(HOW, f"{HOW.stat().st_size // 1024} KB")
    for out, png in sheets.items():
        img = Image.open(io.BytesIO(png))
        if out.name.startswith("badge-"):  # kept at 2x, shown at half in the README: sharp on any screen
            img.save(out, "WEBP", lossless=True, method=6)
        else:
            img.resize((img.width // 2, img.height // 2), Image.LANCZOS).save(out, "WEBP", quality=88, method=6)
        print(out, f"{out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chromium", help="a Chromium binary, if Playwright's own isn't installed")
    asyncio.run(main(ap.parse_args().chromium))
