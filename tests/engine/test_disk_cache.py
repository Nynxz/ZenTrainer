"""dataset.disk_cache: each picture's latents and its own (grounded) conditioning in a file of its
own, read when a step uses it; plain captions only where something still needs them."""

from pathlib import Path

import torch
from PIL import Image

from zentrainer.harness import data
from zentrainer.spec import DatasetSpec


class FakeVAE:
    def encode(self, px):
        return px.movedim(-1, 1)[:, :, ::8, ::8].repeat(1, 2, 1, 1).float()  # (1, 6, h/8, w/8)


class FakeTokenizer:
    llama_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class FakeClip:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def tokenize(self, text, images=(), **kw):
        self.calls.append((text, len(images)))
        return text, len(images)

    def encode_from_tokens_scheduled(self, tokens):
        text, n = tokens
        return [[torch.full((1, 3, 4), float(len(text) + 100 * n)), {"pooled_output": None}]]


def _dataset(root: Path) -> Path:
    ds = root / "pairs"
    (ds / "_controls" / "input").mkdir(parents=True)
    for name, color in (("a", "red"), ("b", "blue")):
        Image.new("RGB", (64, 64), color).save(ds / f"{name}.png")
        (ds / f"{name}.txt").write_text(f"make {name} green")
        Image.new("RGB", (64, 64), "white").save(ds / "_controls" / "input" / f"{name}.png")
    Image.new("RGB", (64, 64), "gray").save(ds / "plain.png")  # no input: trains on its plain caption
    (ds / "plain.txt").write_text("a gray square")
    return ds


def test_disk_samples_read_from_their_files_and_skip_unneeded_captions(tmp_path):
    root = _dataset(tmp_path)
    spec = DatasetSpec(folder="pairs", resolution=64, buckets=False, edit=True, control_sets=["input"],
                       edit_grounded=True, disk_cache=True)
    clip = FakeClip()
    ds = data.build(spec, clip, FakeVAE(), root, tmp_path / "cache", "enc")

    assert all(isinstance(s, data.DiskSample) for s in ds.samples)
    by_caption = {ds.captions[s.caption]: s for s in ds.samples}
    pair, plain = by_caption["make a green"], by_caption["a gray square"]
    assert data.has_inputs(pair) and not data.has_inputs(plain)
    assert pair.latent.dtype == torch.bfloat16 and pair.latent.shape == pair.shape == (6, 8, 8)
    assert len(pair.refs) == 1 and pair.cond[0][0][0, 0, 0].item() == len("make a green") + 100
    assert plain.refs is None and plain.cond is None
    # the grounded pairs' plain captions were never encoded; the plain picture's and "" were
    plain_texts = {t for t, n in clip.calls if n == 0}
    assert plain_texts == {"a gray square", ""}
    assert ds.conditionings[pair.caption] is None and ds.conditionings[plain.caption] is not None
    assert sum(len(v) for v in ds.buckets.values()) == 3

    again = data.build(spec, FakeClip(), FakeVAE(), root, tmp_path / "cache", "enc")  # the cache, read back
    assert [s.file for s in again.samples] == [s.file for s in ds.samples]
    assert torch.equal(again.samples[0].latent, ds.samples[0].latent)


def test_a_folders_caption_sets_are_read_from_one_listing(tmp_path):
    for name, text in (("a.txt", "plain"), ("a.nl.txt", "natural"), ("a.b.txt", "a.b's own"),
                       ("a.b.nl.txt", "a.b natural"), ("a.Bad.txt", "not a set name")):
        (tmp_path / name).write_text(text)
    for img in ("a.png", "a.b.png"):
        Image.new("RGB", (8, 8)).save(tmp_path / img)
    index = data.caption_index([p.name for p in tmp_path.iterdir()])
    for img in ("a.png", "a.b.png"):
        p = tmp_path / img
        assert data.caption_sets(p, ".txt", index) == data.caption_sets(p, ".txt")  # the same as globbing
    assert data.caption_sets(tmp_path / "a.b.png", ".txt", index) == {"default": "a.b's own", "nl": "a.b natural"}


class CountingVAE(FakeVAE):
    def __init__(self):
        self.pictures = 0

    def encode(self, px):
        self.pictures += px.shape[0]
        return super().encode(px)


def test_a_disk_build_cut_short_carries_on_from_its_files(tmp_path):
    root = _dataset(tmp_path)
    for name in ("c", "d", "e"):  # another shape too: batches are one size each
        Image.new("RGB", (96, 64), "green").save(root / f"{name}.png")
        (root / f"{name}.txt").write_text(f"make {name} red")
        Image.new("RGB", (96, 64), "black").save(root / "_controls" / "input" / f"{name}.png")
    spec = DatasetSpec(folder="pairs", resolution=64, buckets=True, bucket_step=16, edit=True, control_sets=["input"],
                       edit_grounded=True, disk_cache=True)
    vae, clip = CountingVAE(), FakeClip()
    ds = data.build(spec, clip, vae, root, tmp_path / "cache", "enc")
    assert vae.pictures == 6 + 5 and sum(n for _, n in clip.calls) == 5  # 6 pictures and 5 inputs; 5 grounded
    first = {s.file: torch.load(s.file, weights_only=False) for s in ds.samples}

    folder = Path(ds.samples[0].file).parent
    (folder.parent / "dataset.pt").unlink()  # cut short: no dataset.pt, one picture never written,
    Path(ds.samples[1].file).unlink()  # one written but never grounded
    kept = torch.load(ds.samples[3].file, weights_only=False)
    kept.pop("cond")
    torch.save(kept, ds.samples[3].file)
    vae, clip = CountingVAE(), FakeClip()
    again = data.build(spec, clip, vae, root, tmp_path / "cache", "enc")
    assert vae.pictures == 2  # the missing picture and its input
    assert sorted(t for t, n in clip.calls if n) == sorted([ds.captions[ds.samples[1].caption], ds.captions[ds.samples[3].caption]])
    for s in again.samples:
        assert torch.equal(s.latent, first[s.file]["latent"]) and data.has_inputs(s) == (s.refs is not None)
        assert (s.cond is None) == (first[s.file].get("cond") is None)


class VideoVAE:
    """As ComfyUI's Wan / Qwen Image VAE: latent_dim 3, and a batch of pictures is one clip's frames
    (one latent back, its frames squeezed 4 to 1 after the first)."""
    latent_dim, not_video = 3, False

    def __init__(self):
        self.calls = 0

    def encode(self, px):
        self.calls += 1
        frames = px.movedim(-1, 1)[:, :, ::8, ::8].repeat(1, 2, 1, 1).float()  # (T, 6, h/8, w/8)
        return frames[::4].movedim(0, 1)[None]  # (1, 6, 1 + (T - 1) // 4, h/8, w/8)


def test_a_video_vae_encodes_each_cached_picture_as_its_own(tmp_path):
    # Krea 2 on Qwen Image's VAE with the latent cache on: the 3 pictures of one size used to go in one call,
    # come back as one clip's latent, and the second picture found none (IndexError)
    root = _dataset(tmp_path)
    spec = DatasetSpec(folder="pairs", resolution=64, buckets=False, disk_cache=True)
    vae = VideoVAE()
    ds = data.build(spec, FakeClip(), vae, root, tmp_path / "cache", "enc")
    assert vae.calls == 3 and len(ds.samples) == 3
    latents = {ds.captions[s.caption]: s.latent for s in ds.samples}
    assert all(lat.shape == (6, 1, 8, 8) for lat in latents.values())  # one frame each, as _encode gives
    assert not torch.equal(latents["make a green"], latents["make b green"])  # each its own picture (red, blue)

    # a cache the failing build left: a picture holding a batch squeezed into one clip (more than one
    # frame), whole and readable; it's encoded again, not kept
    s = ds.samples[0]
    bad = torch.load(s.file, weights_only=False)
    bad["latent"] = bad["latent"].repeat(1, 2, 1, 1)
    torch.save(bad, s.file)
    (Path(s.file).parent.parent / "dataset.pt").unlink()
    vae = VideoVAE()
    again = data.build(spec, FakeClip(), vae, root, tmp_path / "cache", "enc")
    assert vae.calls == 1 and all(t.latent.shape == (6, 1, 8, 8) for t in again.samples)


def test_preserve_encodes_each_caption_with_its_class_words_and_keys_the_cache(tmp_path):
    root = tmp_path / "jane"
    root.mkdir()
    Image.new("RGB", (64, 64), "red").save(root / "a.png")
    (root / "a.txt").write_text("[trigger] standing in a kitchen")
    spec = DatasetSpec(folder="jane", resolution=64, buckets=False, trigger="JaneDoe")
    clip = FakeClip()
    ds = data.build(spec, clip, FakeVAE(), root, tmp_path / "cache", "enc", preserve=("woman", "man"))
    assert ds.captions[ds.samples[0].caption] == "JaneDoe standing in a kitchen"
    assert {"woman standing in a kitchen", "man standing in a kitchen"} <= set(ds.captions)  # encoded with the rest
    assert {"a photo of a woman", "a photo of a man"} <= set(ds.captions)  # preserve_plain's prompts too
    assert all(ds.conditionings[ds.captions.index(c)] is not None for c in ds.captions)
    plain = data.build(spec, FakeClip(), FakeVAE(), root, tmp_path / "cache", "enc")  # without: its own cache
    assert "woman standing in a kitchen" not in plain.captions
    assert data.as_class("janedoe, JaneDoe's coat; notjanedoe", "JaneDoe", "woman") == "woman, woman's coat; notjanedoe"
