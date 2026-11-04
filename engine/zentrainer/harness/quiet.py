"""ComfyUI's model-load log line, once while a run works through many loads."""

from __future__ import annotations

import contextlib
import os


@contextlib.contextmanager
def quiet_reloads():
    """While a run encodes its dataset or text, or renders samples. ComfyUI logs a dynamic model's
    "prepared for dynamic VRAM loading" at every load, and an unchanged repeat only at DEBUG while
    a tqdm bar is up (as its sampler's is): one that draws nowhere keeps the text encoder's and
    VAE's line to once, not once a caption or picture, with nothing new in the console or the
    app's log."""
    import tqdm

    with open(os.devnull, "w") as nowhere:
        bar = tqdm.tqdm(total=1, file=nowhere, leave=False)
        try:
            yield
        finally:
            bar.close()
