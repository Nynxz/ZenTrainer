"""The web app's live views don't animate progress (see web/src/app.css): a width or
stroke-dasharray transition on something a running job updates keeps the page repainting for
the whole run, on the GPU training uses."""

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[2] / "web" / "src"
# transitions on things only a person moves: (file, selector)
ALLOWED = {
    ("components/AppSidebar.vue", ".rail"),  # the sidebar folding
    ("components/ScrollArea.vue", ".thumb"),  # the scrollbar widening under the pointer
}
RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")


def test_nothing_animates_a_progress_change():
    found = []
    for f in sorted(WEB.rglob("*")):
        if f.suffix not in (".vue", ".css"):
            continue
        text = f.read_text()
        if f.suffix == ".vue":
            text = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", text, re.S))
        for sel, body in RULE.findall(text):
            if re.search(r"transition(-property)?\s*:[^;]*\b(width|stroke-dasharray|all)\b", body):
                key = (f.relative_to(WEB).as_posix(), sel.strip().splitlines()[-1].strip())
                if key not in ALLOWED:
                    found.append(key)
    assert not found, f"transitions that animate a progress change (a running job's repaints): {found}"
