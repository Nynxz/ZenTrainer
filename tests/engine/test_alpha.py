"""Transparent pictures: a four-channel VAE (Qwen Image 2.1's) gets their alpha, in targets, edit
inputs and its own cache; vision encoders see them over white; slider strips keep it."""

from pathlib import Path

import torch
from PIL import Image

from zentrainer.harness import data
from zentrainer.harness.callbacks import _strip
from zentrainer.harness.data.edit import grounding_pixels
from zentrainer.spec import DatasetSpec


class VAE:
    def __init__(self, channels: int = 3):
        self.output_channels = channels

    def encode(self, px):
        return px.movedim(-1, 1)[:, :, ::8, ::8].float()  # (1, C, h/8, w/8): the pixels' channels kept


class Clip:
    def tokenize(self, text, **kw):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 3, 4), {"pooled_output": None}]]


def _see_through(path: Path, alpha: int = 0) -> None:
    Image.new("RGBA", (64, 64), (200, 100, 50, alpha)).save(path)


def test_rgba_loads_keep_alpha_and_rgb_files_come_opaque(tmp_path):
    _see_through(tmp_path / "t.png", 128)
    Image.new("RGB", (64, 64), "red").save(tmp_path / "o.png")
    assert data.pixel_mode(VAE(4)) == "RGBA" and data.pixel_mode(VAE()) == "RGB" and data.pixel_mode(None) == "RGB"
    px = data.load_image(tmp_path / "t.png", (32, 32), "RGBA")
    assert px.shape == (32, 32, 4) and abs(px[..., 3].mean().item() - 128 / 255) < 1e-3
    assert torch.all(data.load_image(tmp_path / "o.png", (32, 32), "RGBA")[..., 3] == 1)
    assert data.load_image(tmp_path / "t.png", (32, 32)).shape == (32, 32, 3)


def test_a_four_channel_vae_trains_on_alpha_with_its_own_cache(tmp_path):
    ds = tmp_path / "pairs"
    (ds / "_controls" / "input").mkdir(parents=True)
    _see_through(ds / "a.png")  # the cut-out: all transparent
    (ds / "a.txt").write_text("isolate the subject")
    Image.new("RGB", (64, 64), "white").save(ds / "_controls" / "input" / "a.png")
    spec = DatasetSpec(folder="pairs", resolution=64, buckets=False, edit=True, control_sets=["input"])

    rgba = data.build(spec, Clip(), VAE(4), ds, tmp_path / "cache", "enc")
    s = rgba.samples[0]
    assert s.latent.shape[0] == 4 and torch.all(s.latent[3] == 0)  # target: transparent
    assert s.refs[0].shape[0] == 4 and torch.all(s.refs[0][3] == 1)  # input: an opaque RGB file
    rgb = data.build(spec, Clip(), VAE(), ds, tmp_path / "cache", "enc")  # same VAE name, RGB: not that cache
    assert rgb.samples[0].latent.shape[0] == 3
    assert len([p for p in (tmp_path / "cache").iterdir() if (p / "dataset.pt").is_file()]) == 2


def test_the_vision_encoder_sees_transparency_over_white(tmp_path):
    _see_through(tmp_path / "t.png", 0)
    px = grounding_pixels(tmp_path / "t.png")
    assert px.shape[-1] == 3 and torch.all(px == 1)


def test_a_slider_strip_keeps_transparency():
    shots = [torch.zeros(16, 16, 4), torch.ones(16, 16, 4)]
    out = _strip(shots, [-1, 1])
    assert out.mode == "RGBA" and out.getpixel((0, out.height - 1))[3] == 0
    assert _strip([torch.zeros(16, 16, 3)] * 2, [-1, 1]).mode == "RGB"
