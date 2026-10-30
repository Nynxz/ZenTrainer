"""Model families from ComfyUI templates: parsing, grouping, recommendations. No ComfyUI needed:
a tiny fake templates folder stands in for the comfyui_workflow_templates package."""

import json

import pytest

from zentrainer.models import families
from zentrainer.models.families import Family, ModelFile


def _split_template(unet, clip, clip_type, vae, repo, ksampler=None):
    hf = lambda folder, name: {"name": name, "directory": folder,
                               "url": f"https://huggingface.co/{repo}/resolve/main/split_files/{folder}/{name}"}
    nodes = [
        {"type": "UNETLoader", "widgets_values": [unet, "default"], "properties": {"models": [hf("diffusion_models", unet)]}},
        {"type": "CLIPLoader", "widgets_values": [clip, clip_type, "default"], "properties": {"models": [hf("text_encoders", clip)]}},
        {"type": "VAELoader", "widgets_values": [vae], "properties": {"models": [hf("vae", vae)]}},
    ]
    if ksampler:
        nodes.append({"type": "KSampler", "widgets_values": [0, "fixed", *ksampler, 1.0]})
    # loaders inside a subgraph are found too
    return {"nodes": nodes[:1], "definitions": {"subgraphs": [{"nodes": nodes[1:]}]}}


@pytest.fixture
def templates(tmp_path, monkeypatch):
    root = tmp_path / "templates"
    root.mkdir()
    entries = {
        "krea_turbo": (["Krea 2"], _split_template("krea2_turbo_fp8_scaled.safetensors", "qwen3vl_4b.safetensors", "krea2",
                                                  "qwen_image_vae.safetensors", "Comfy-Org/Krea-2", [8, 1.0, "euler", "simple"])),
        # names a partner model too: grouped with Krea by its repo, not split off
        "krea_plus_api": (["Kling 2.6", "Krea 2"], _split_template("krea2_turbo_fp8_scaled.safetensors", "qwen3vl_4b.safetensors",
                                                                  "krea2", "qwen_image_vae.safetensors", "Comfy-Org/Krea-2")),
        "sdxl": (["SDXL", "Stability"], {"nodes": [{"type": "CheckpointLoaderSimple", "widgets_values": ["sd_xl_base_1.0.safetensors"],
                 "properties": {"models": [{"name": "sd_xl_base_1.0.safetensors", "directory": "checkpoints",
                                            "url": "https://huggingface.co/stabilityai/sdxl/resolve/main/sd_xl_base_1.0.safetensors"}]}}]}),
        "sd3": (["SD3.5", "Stability"], {"nodes": [{"type": "CheckpointLoaderSimple", "widgets_values": ["sd3.5_large.safetensors"],
                "properties": {"models": [{"name": "sd3.5_large.safetensors", "directory": "checkpoints",
                                           "url": "https://huggingface.co/sai/sd3/resolve/main/sd3.5_large.safetensors"}]}}]}),
        # two diffusion models (base + refiner style): not a shape the harness trains
        "two_unets": (["Odd"], {"nodes": [{"type": "UNETLoader", "widgets_values": ["a.safetensors"]},
                                          {"type": "UNETLoader", "widgets_values": ["b.safetensors"]}]}),
    }
    index = [{"templates": [{"name": k, "mediaType": "image", "tags": ["Text to Image"], "models": m} for k, (m, _) in entries.items()]
             + [{"name": "api_x", "mediaType": "image", "tags": ["Text to Image"], "models": ["Kling 2.6"]}]}]
    (root / "index.json").write_text(json.dumps(index))
    for k, (_, w) in entries.items():
        (root / f"{k}.json").write_text(json.dumps(w))
    monkeypatch.setattr(families, "_templates_dir", lambda: root)
    return root


def test_families_from_templates(templates):
    fams = families._from_templates()
    assert set(fams) == {"krea-2", "sdxl", "sd3-5"}  # Stability split by model; the two-unet template skipped
    krea = fams["krea-2"]
    assert (krea.kind, krea.clip_type, krea.repos) == ("split", "krea2", ["Comfy-Org/Krea-2"])
    assert sorted(krea.templates) == ["krea_plus_api", "krea_turbo"]
    assert [f.name for f in krea.files["diffusion_models"]] == ["krea2_turbo_fp8_scaled.safetensors"]
    assert krea.files["diffusion_models"][0].distilled
    assert krea.sample_defaults == {"model": "krea2_turbo_fp8_scaled.safetensors", "steps": 8, "cfg": 1.0,
                                    "sampler": "euler", "scheduler": "simple"}


def test_family_name_is_the_most_specific():
    freq = {"Flux": 40, "Flux.2 Dev": 3, "SDXL": 6, "Stability": 30}
    assert families._family_name(["Flux", "Flux.2 Dev"], freq) == "Flux.2 Dev"
    assert families._family_name(["SDXL", "Stability"], freq) == "SDXL"


def _fam(*unets):
    f = Family("krea-2", "Krea 2", "split", "krea2", ["Comfy-Org/Krea-2"], ["t"])
    f.files = {
        "diffusion_models": [ModelFile(n, "diffusion_models", "u", distilled="turbo" in n, from_template=(i == 0), installed=inst)
                             for i, (n, inst) in enumerate(unets)],
        "text_encoders": [ModelFile("qwen3vl_4b.safetensors", "text_encoders", "u", from_template=True, installed="qwen3vl_4b.safetensors")],
        "vae": [ModelFile("qwen_image_vae.safetensors", "vae", "u", from_template=True)],
    }
    f.sample_defaults = {"model": unets[0][0], "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"}
    return f


def test_recommend_trains_on_base_and_samples_on_turbo():
    # nothing installed: the base build in the template's precision
    f = _fam(("krea2_turbo_fp8_scaled.safetensors", None), ("krea2_raw_bf16.safetensors", None), ("krea2_raw_fp8_scaled.safetensors", None))
    rec = families.recommend(f)
    assert rec["train"]["diffusion_models"] == "krea2_raw_fp8_scaled.safetensors"
    assert rec["sample"]["model"] == "krea2_turbo_fp8_scaled.safetensors"
    assert [m["name"] for m in families.missing(f, rec["train"])] == ["krea2_raw_fp8_scaled.safetensors", "qwen_image_vae.safetensors"]
    # an installed base build wins over precision
    f = _fam(("krea2_turbo_fp8_scaled.safetensors", None), ("krea2_raw_bf16.safetensors", "sub/krea2_raw_bf16.safetensors"),
             ("krea2_raw_fp8_scaled.safetensors", None))
    assert families.recommend(f)["train"]["diffusion_models"] == "krea2_raw_bf16.safetensors"
    # only the turbo build exists: train on it, no separate sampling model to suggest
    f = _fam(("krea2_turbo_fp8_scaled.safetensors", None))
    rec = families.recommend(f)
    assert rec["train"]["diffusion_models"] == "krea2_turbo_fp8_scaled.safetensors" and rec["sample"] is None


def test_downloads_only_catalog_files(templates, monkeypatch):
    from zentrainer.models import downloads

    monkeypatch.setattr(families, "_families", None)
    with pytest.raises(ValueError, match="isn't a"):
        downloads.Downloads().start("krea-2", "vae", "../../etc/passwd")
    with pytest.raises(ValueError, match="isn't a"):
        downloads.Downloads().start("nope", "vae", "qwen_image_vae.safetensors")


TYPES = ["stable_diffusion", "sd3", "lumina2", "hidream", "qwen_image", "flux2", "ovis", "longcat_image", "krea2", "moss_image"]


def test_clip_type_from_repo_name():
    assert families.match_clip_type("MOSS-Image", TYPES) == "moss_image"
    assert families.match_clip_type("Qwen-Image_ComfyUI", TYPES) == "qwen_image"
    assert families.match_clip_type("LongCat-Image", TYPES) == "longcat_image"
    assert families.match_clip_type("Krea-2", TYPES) == "krea2"
    assert families.match_clip_type("HiDream-I1_ComfyUI", TYPES) == "hidream"  # longest type the name starts with
    assert families.match_clip_type("z_image", TYPES) is None  # no guess when nothing matches


def test_family_from_comfy_org_repo():
    rows = [{"path": f"{d}/{n}", "size": 1} for d, n in [
        ("diffusion_models", "moss_image_sft_bf16.safetensors"), ("diffusion_models", "moss_image_base_bf16.safetensors"),
        ("text_encoders", "qwen3.5_9b_int8_convrot.safetensors"), ("vae", "flux2-vae.safetensors")]] + [{"path": "README.md"}]
    fam = families._family_from_repo("Comfy-Org/MOSS-Image", rows, TYPES)
    assert (fam.id, fam.name, fam.kind, fam.clip_type, fam.source) == ("moss-image", "MOSS Image", "split", "moss_image", "repo")
    assert families.recommend(fam)["train"]["diffusion_models"] == "moss_image_base_bf16.safetensors"  # base over sft
    # bare files at the root are checkpoints
    ck = families._family_from_repo("Comfy-Org/stable-diffusion-v1-5-archive", [{"path": "v1-5-pruned-emaonly.safetensors"}], TYPES)
    assert (ck.kind, ck.name, list(ck.files)) == ("checkpoint", "Stable Diffusion v1.5", ["checkpoints"])
    # a split repo whose text encoder type can't be told: skipped, not guessed
    assert families._family_from_repo("Comfy-Org/Mystery", rows, TYPES) is None


def test_family_from_maker_comfy_repo():
    # Flux 3 Klein: the maker's own ComfyUI-layout repo, gated, with a prompt upsampler LLM beside the text encoder
    rows = [{"path": f"{d}/{n}", "size": 1} for d, n in [
        ("diffusion_models", "flux3-image-klein-bf16.safetensors"), ("diffusion_models", "flux3-image-klein-base-bf16.safetensors"),
        ("text_encoders", "qwen3.5_9b_flux3_prompt_upsampler_bf16.safetensors"), ("text_encoders", "qwen3vl_4b_bf16.safetensors"),
        ("vae", "flux2-vae.safetensors")]]
    fam = families._family_from_repo("black-forest-labs/flux-3-image-klein-comfy", rows, TYPES + ["flux3"],
                                     {"created": "2026-10-14", "gated": True})
    assert (fam.name, fam.clip_type, fam.added, fam.gated) == ("Flux 3 Image Klein", "flux3", "2026-10-14", True)
    assert [f.name for f in fam.files["text_encoders"]] == ["qwen3vl_4b_bf16.safetensors"]
    assert families.recommend(fam)["train"]["diffusion_models"] == "flux3-image-klein-base-bf16.safetensors"


def test_flux3_template_with_prompt_upsampler(tmp_path, monkeypatch):
    # two CLIPLoaders: the prompt upsampler LLM and the real text encoder; the "distilled" template's
    # model has no turbo-ish name, and other precisions of it listed from the repo follow it
    repo = "black-forest-labs/flux-3-image-klein-comfy"
    hf = lambda folder, name: {"name": name, "directory": folder, "url": f"https://huggingface.co/{repo}/resolve/main/{folder}/{name}"}
    def tmpl(unet):
        return {"nodes": [
            {"type": "CLIPLoader", "widgets_values": ["qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors", "flux3", "default"],
             "properties": {"models": [hf("text_encoders", "qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors")]}},
            {"type": "UNETLoader", "widgets_values": [unet, "default"], "properties": {"models": [hf("diffusion_models", unet)]}},
            {"type": "CLIPLoader", "widgets_values": ["qwen3vl_4b_int8_convrot.safetensors", "flux3", "default"],
             "properties": {"models": [hf("text_encoders", "qwen3vl_4b_int8_convrot.safetensors")]}},
            {"type": "VAELoader", "widgets_values": ["flux2-vae.safetensors"], "properties": {"models": [hf("vae", "flux2-vae.safetensors")]}},
            {"type": "KSampler", "widgets_values": [3, "randomize", 15, 1, "er_sde", "simple", 1]}]}
    root = tmp_path / "templates"
    root.mkdir()
    entries = {"image_flux3_klein_t2i": "flux3-image-klein-base-int8-convrot.safetensors",
               "image_flux3_klein_distilled_t2i": "flux3-image-klein-int8-convrot.safetensors"}
    (root / "index.json").write_text(json.dumps([{"templates": [
        {"name": k, "mediaType": "image", "tags": ["Text to Image"], "models": ["FLUX 3 Image"], "date": "2026-10-15"} for k in entries]}]))
    for k, unet in entries.items():
        (root / f"{k}.json").write_text(json.dumps(tmpl(unet)))
    monkeypatch.setattr(families, "_templates_dir", lambda: root)
    fam = families._from_templates()["flux-3-image"]
    assert (fam.clip_type, fam.added) == ("flux3", "2026-10-15")
    assert [f.name for f in fam.files["text_encoders"]] == ["qwen3vl_4b_int8_convrot.safetensors"]
    assert fam.sample_defaults["model"] == "flux3-image-klein-int8-convrot.safetensors"
    monkeypatch.setattr(families, "repo_files", lambda *a: [{"path": "diffusion_models/flux3-image-klein-bf16.safetensors"},
                                                            {"path": "diffusion_models/flux3-image-klein-base-bf16.safetensors"}])
    families._add_repo_files(fam)
    assert {f.name: f.distilled for f in fam.files["diffusion_models"]} == {
        "flux3-image-klein-base-int8-convrot.safetensors": False, "flux3-image-klein-int8-convrot.safetensors": True,
        "flux3-image-klein-bf16.safetensors": True, "flux3-image-klein-base-bf16.safetensors": False}
    rec = families.recommend(fam)
    assert rec["train"]["diffusion_models"] == "flux3-image-klein-base-int8-convrot.safetensors"  # the template's precision
    assert rec["sample"]["model"] == "flux3-image-klein-int8-convrot.safetensors"


def test_profiles_load_and_validate(tmp_path):
    from zentrainer.models import profiles

    shipped = profiles.load()
    moss = shipped["moss-image"]
    assert moss["status"] == "recipe" and moss["samples"]["slg_layers"] == [22] and moss["sources"]
    (tmp_path / "bad.json").write_text(json.dumps({"families": ["x"], "status": "recipe", "samples": {"nope": 1}}))
    (tmp_path / "ok.json").write_text(json.dumps({"families": ["a", "b"], "status": "verified", "samples": {"steps": 8}}))
    got = profiles.load(tmp_path)
    assert set(got) == {"a", "b"} and got["a"]["id"] == "ok" and got["a"]["notes"] == []  # bad one skipped, defaults filled


def test_only_families_trained_here_are_tested(monkeypatch):
    # families.json's `tested`: those are listed as models; one the catalog finds that nobody has
    # trained (a new template, a new repo) is untested, under Other models
    cat = families._catalog()
    assert cat["tested"] and len(set(cat["tested"])) == len(cat["tested"])
    for fid, want in (("moss-image", True), ("some-new-model", False)):
        assert Family(fid, fid, "split", None, [], []).public()["tested"] is want
    # what ZenTrainer can't train at all is left out of the catalog
    assert families.correction("ming-3-0-tts").get("hide")
