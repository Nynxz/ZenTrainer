"""A text encoder short of memory can hand back NaN: never cached, and one cached before the check
encoded again (data.finite / checked_encoding, TextCache, Samples' prompt and baseline caches)."""

import types

import pytest
import torch
from PIL import Image

from zentrainer.harness import data
from zentrainer.harness.callbacks import Samples

NAN = [[torch.full((1, 3, 4), float("nan")), {"pooled_output": None}]]
OK = [[torch.ones(1, 3, 4), {"pooled_output": torch.zeros(1, 2)}]]


def test_finite_looks_into_nested_conditionings():
    assert data.finite(OK)
    assert not data.finite(NAN)
    assert not data.finite([[torch.ones(1), {"pooled_output": torch.tensor([float("inf")])}]])
    assert data.finite([[torch.ones(1, dtype=torch.long), {"attention_mask": None}]])


def test_a_nan_caption_encoding_is_never_cached(tmp_path):
    texts = data.TextCache(tmp_path, "enc")
    with pytest.raises(ValueError, match="ran out of memory"):
        texts.put("a cat", NAN)
    assert not texts.dir.exists() or not any(texts.dir.iterdir())


def test_a_nan_encoding_cached_before_the_check_is_encoded_again(tmp_path):
    texts = data.TextCache(tmp_path, "enc")
    texts.put("a cat", OK)
    f = texts.dir / f"{texts.key('a cat')}.pt"
    torch.save(NAN, f)  # as an older build cached it
    assert texts.get("a cat") is None
    assert not f.exists()


class _Clip:
    def __init__(self, value):
        self.value, self.calls = value, 0

    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, _tokens):
        self.calls += 1
        return [[torch.full((1, 3, 4), self.value), {}]]


def _run(tmp_path, clip):
    return types.SimpleNamespace(encoder_id="enc", cache_dir=tmp_path, models=types.SimpleNamespace(clip=clip))


def test_sample_prompts_never_cache_nan(tmp_path):
    t = _run(tmp_path, _Clip(float("nan")))
    with pytest.raises(ValueError, match="ran out of memory"):
        Samples(1)._encode_texts(t, ["a cat"], "")
    assert not list(tmp_path.rglob("*.pt"))


def test_sample_prompts_cached_as_nan_are_encoded_again(tmp_path):
    good = _Clip(1.0)
    t = _run(tmp_path, good)
    Samples(1)._encode_texts(t, ["a cat"], "")
    (path,) = tmp_path.rglob("*.pt")
    torch.save({"prompts": NAN, "negative": NAN}, path)  # as an older build cached it
    conds = Samples(1)._encode_texts(t, ["a cat"], "")
    assert good.calls == 4  # prompt and negative, encoded twice
    assert data.finite(conds)


def test_a_black_baseline_is_not_reused(tmp_path):
    s = Samples(1)
    s._baseline_dir = lambda: tmp_path
    s._baseline_key = lambda _t, i: f"key{i}"
    Image.new("RGB", (8, 8)).save(tmp_path / "key0.png")  # a NaN render, kept as black
    Image.new("RGB", (8, 8), (20, 30, 40)).save(tmp_path / "key1.png")
    assert s._baseline_file(None, 0) is None
    assert not (tmp_path / "key0.png").exists()
    assert s._baseline_file(None, 1) == tmp_path / "key1.png"


def test_an_older_cache_with_a_nan_caption_inside_is_rebuilt():
    d = {"latents": [torch.zeros(4, 8, 8)], "caption_idx": [0], "captions": ["a cat"], "empty_caption": 0,
         "conditionings": [NAN]}
    with pytest.raises(ValueError, match="NaN"):
        data._from_plain(d)
    d["conditionings"] = [OK]
    assert len(data._from_plain(d).samples) == 1


def test_a_black_video_baseline_is_not_reused(tmp_path):
    from zentrainer.harness.data import video

    s = Samples(1)
    s._baseline_dir = lambda: tmp_path
    s._baseline_key = lambda _t, i: f"key{i}"
    video.write(tmp_path / "key0.mp4", torch.zeros(3, 16, 16, 3), 8)
    video.write(tmp_path / "key1.mp4", torch.full((3, 16, 16, 3), 0.5), 8)
    assert s._baseline_file(None, 0) is None
    assert s._baseline_file(None, 1) == tmp_path / "key1.mp4"


def test_has_counts_only_usable_encodings(tmp_path):
    texts = data.TextCache(tmp_path, "enc")
    texts.put("a cat", OK)
    assert texts.has("a cat")
    torch.save(NAN, texts.dir / f"{texts.key('a cat')}.pt")
    assert not texts.has("a cat")


def test_a_disk_sample_with_a_nan_encoding_says_to_clear_the_cache(tmp_path):
    f = tmp_path / "s.pt"
    torch.save({"latent": torch.zeros(4, 8, 8), "cond": NAN}, f)
    sample = data.DiskSample(f, (4, 8, 8), 0, [0], False, True)
    assert sample.latent.shape == (4, 8, 8)
    with pytest.raises(ValueError, match="clear this dataset's cache"):
        sample.cond
