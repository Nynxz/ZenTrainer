"""Model families: which files make up a model, whether they're installed, where to get them.

Built from ComfyUI's own workflow templates (the comfyui_workflow_templates package that ships
with ComfyUI). Each open text-to-image template names its loaders' files, the folder each goes
in and a download URL, so a family here is "the files those templates load", grouped by the
model name the template index gives. Nothing is hand-maintained: a new ComfyUI brings new
families.

Templates are inference workflows, so they mostly load the turbo/distilled build. Training
wants the base, so a family can also list every build in its Hugging Face repo (fetched on
request, cached for a day): `krea2_raw_*` beside `krea2_turbo_*`.

ComfyUI supports more models than it has templates for (MOSS Image had none when it landed,
Flux 3 Klein shipped from its maker's own repo), so Hugging Face repos are a second source:
every Comfy-Org repo, and the ComfyUI-layout repos ("...-comfy") model makers publish. A repo
laid out as diffusion model + text encoder + VAE (or as checkpoints) becomes a family, its
text encoder type matched by name against ComfyUI's CLIPLoader types. These carry
`source: "repo"`.

Each family has an `added` date (its first template's, or the repo's creation) so the newest
models can sort first.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("zentrainer")

# ComfyUI folder for a repo directory name (the directory a file sits in)
FOLDERS = {
    "diffusion_models": "diffusion_models", "unet": "diffusion_models",
    "text_encoders": "text_encoders", "clip": "text_encoders",
    "vae": "vae", "checkpoints": "checkpoints",
}
PARTS = ("diffusion_models", "text_encoders", "vae")  # what a split model needs, in spec order
DISTILLED = re.compile(r"turbo|distill|lightning|hyper|lcm|schnell|\bfast\b", re.I)
BASE = re.compile(r"(^|[_-])(base|raw)([_.-]|$)", re.I)  # among non-distilled builds, the one to train on
# repos that aren't text-to-image models: video, audio/music, 3D, upscalers, edit/control add-ons
NOT_IMAGE = re.compile(
    r"video|audio|music|wan_?2|ltx|hunyuan_?video|hunyuan_?3d|3d|cosmos|mochi|svd|ace_step|stable_audio|yue|"
    r"sam|seedvr|pose|lotus|controlnet|redux|kontext|edit|upscal|esrgan|vision|sigclip|layered|depth", re.I)
# video models the harness trains (on images as one-frame clips; see harness/data/streams.py): their
# text-to-video template defines the family
TRAINABLE_VIDEO = re.compile(r"minimax_h3", re.I)
ORG = "Comfy-Org"
# model makers who publish ComfyUI-layout repos themselves; only their repos named "comfy" count
MAKERS = ("black-forest-labs", "stabilityai", "Qwen", "Tongyi-MAI", "krea", "OpenMOSS", "HiDream-ai",
          "meituan-longcat", "tencent", "Lightricks")
# text encoder folder files that aren't the text encoder (Flux 3 Klein ships a prompt upsampler LLM)
NOT_PICTURE_VAE = re.compile(r"audio", re.I)
CATALOG = Path(__file__).resolve().parents[2] / "catalog" / "families.json"  # hand corrections (see its _about)


def _catalog() -> dict:
    try:
        return json.loads(CATALOG.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("[ZenTrainer] can't read %s: %s", CATALOG, e)
        return {}


# text encoder files that aren't the text encoder (Flux 3 Klein's prompt upsampler, Qwen-Image 2.1's
# prompt enhancer "_pe_"): the catalog's text_encoder_skip
NOT_TE = re.compile("|".join(_catalog().get("text_encoder_skip") or ["upsampl"]), re.I)


def correction(fid: str) -> dict:
    """The catalog's entry for a family: maker, logo, mono, name, hide (the first match)."""
    for e in _catalog().get("families") or []:
        if re.search(e.get("match", "^$"), fid):
            return e
    return {}


def _correct(fams: dict[str, "Family"]) -> None:
    """The catalog's renames and hides, applied in place (ids stay, so favourites keep pointing)."""
    for fid in list(fams):
        e = correction(fid)
        if e.get("hide"):
            del fams[fid]
        elif e.get("name"):
            fams[fid].name = e["name"]
HF_URL = re.compile(r"^https://huggingface\.co/([\w.-]+/[\w.-]+)/resolve/([\w.-]+)/(.+)$")
REPO_TTL = 24 * 3600
# which repos the publishers have: shorter, so a model they add shows up the same day (the
# catalog's Re-list refetches it at once)
PUBLISHERS_TTL = 3 * 3600


@dataclass
class ModelFile:
    name: str  # file name, as the repo/template spells it
    folder: str  # ComfyUI folder type
    url: str
    size: int | None = None
    distilled: bool = False
    from_template: bool = False
    installed: str | None = None  # the name ComfyUI's loaders list it by (may include a subfolder)


@dataclass
class Family:
    id: str
    name: str
    kind: str  # "split" | "checkpoint"
    clip_type: str | None
    repos: list[str]
    templates: list[str]
    files: dict[str, list[ModelFile]] = field(default_factory=dict)  # folder -> files
    # sampler settings the templates use with their (mostly turbo) model
    sample_defaults: dict[str, Any] = field(default_factory=dict)
    repo_listed: bool = False
    source: str = "templates"  # "templates" | "repo" (a Hugging Face repo with no template yet)
    added: str | None = None  # ISO date: the first template's, or the repo's creation
    gated: bool = False  # the repo asks you to accept a license before downloading

    def public(self) -> dict:
        from . import profiles

        d = asdict(self)
        e = correction(self.id)
        d["maker"], d["logo"], d["mono"] = e.get("maker"), e.get("logo"), bool(e.get("mono"))
        d["other"] = bool(e.get("other"))  # listed under "Other models", out of the main list
        d["tested"] = self.id in (_catalog().get("tested") or [])  # trained end to end here (see its _about)
        d["profile"] = profiles.for_family(self.id)
        d["recommended"] = recommend(self)
        d["missing"] = missing(self, d["recommended"]["train"])
        return d


def _templates_dir() -> Path | None:
    try:
        import comfyui_workflow_templates_json as pkg  # ships with ComfyUI

        return Path(pkg.__file__).resolve().parent / "templates"
    except ImportError:
        return None


def _nodes(w: dict) -> list[dict]:
    out = list(w.get("nodes", []))
    for sg in w.get("definitions", {}).get("subgraphs", []):
        out += sg.get("nodes", [])
    return out


def _model_urls(o: Any, out: dict[str, dict]) -> None:
    """Every {name, url, directory} a template's nodes declare under properties.models."""
    if isinstance(o, dict):
        ms = o.get("models")
        if isinstance(ms, list):
            for m in ms:
                if isinstance(m, dict) and m.get("url") and m.get("name"):
                    out.setdefault(m["name"], m)
        for v in o.values():
            _model_urls(v, out)
    elif isinstance(o, list):
        for v in o:
            _model_urls(v, out)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _family_name(models: list[str], freq: dict[str, int]) -> str:
    """The most specific of a template's model names: the one fewest templates share.
    ['Flux', 'Flux.2 Dev'] -> 'Flux.2 Dev', ['SDXL', 'Stability'] -> 'SDXL' (the index
    isn't consistent about which comes first)."""
    return min(models, key=lambda m: (freq.get(m, 0), models.index(m))) if models else ""


TASK = re.compile(r"\s*[-:]?\s*(text[ -]to[ -](image|video)|image[ -]to[ -]video|image edit(ing)?|t2i|t2v)\s*$", re.I)


def _title_name(recs: list[dict]) -> str:
    """A family's name from its templates' titles, for when the index gives two models one name
    ("NetaYume Lumina Text to Image" and "OmniGen2: Text to Image", both filed as OmniGen): the
    title before any ":" without the task, the shortest of its templates'."""
    names = [TASK.sub("", r["title"].split(":")[0]).strip() for r in recs]
    return min((n for n in names if n), key=len, default="")


def _from_templates() -> dict[str, Family]:
    root = _templates_dir()
    if root is None or not (root / "index.json").is_file():
        return {}
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    freq: dict[str, int] = {}
    for cat in index:
        for t in cat.get("templates", []):
            for m in t.get("models", []):
                freq[m] = freq.get(m, 0) + 1
    # group templates by what they load: a split model by its diffusion model's repo (the real
    # identity: every Qwen-Image template shares one), a checkpoint by model name
    groups: dict[tuple, list[dict]] = {}
    for cat in index:
        for t in cat.get("templates", []):
            tags = t.get("tags", [])
            image = t.get("mediaType") == "image" and "Text to Image" in tags
            video = bool(TRAINABLE_VIDEO.search(t["name"])) and "Text to Video" in tags
            if t["name"].startswith("api_") or not (image or video):
                continue
            rec = _read_template(root, t, freq)
            if rec is None:
                _unsupported.update(_template_repos(root, t["name"]))  # a shape the harness can't load
            else:
                key = ("split", rec["repo"]) if rec["kind"] == "split" else ("checkpoint", rec["name"])
                groups.setdefault(key, []).append(rec)
    def index_name(recs: list[dict]) -> str:
        counts: dict[str, int] = {}
        for r in recs:
            counts[r["name"]] = counts.get(r["name"], 0) + 1
        return max(counts, key=lambda n: (counts[n], -freq.get(n, 0)))

    # two groups (two repos) under one model name are two models: each named by its titles
    named = {key: index_name(recs) for key, recs in groups.items()}
    shared = {n for n in named.values() if list(named.values()).count(n) > 1}
    for key, recs in groups.items():
        if named[key] in shared and (t := _title_name(recs)):
            named[key] = t
    fams: dict[str, Family] = {}
    for key, recs in groups.items():
        name = named[key]
        first = recs[0]
        fid = _slug(name)
        if any(f.name == name for f in fams.values()) and first["repo"]:
            name = f"{name} ({first['repo'].split('/')[-1]})"  # still the same: tell them apart by repo
            fid = _slug(name)
        while fid in fams:
            fid += "-2"
        fam = Family(fid, name, first["kind"], first["clip_type"], [], [])
        sample_distilled = False
        for r in recs:
            if r["clip_type"] != fam.clip_type:
                continue
            fam.templates.append(r["template"])
            if r["repo"] and r["repo"] not in fam.repos:
                fam.repos.append(r["repo"])
            for folder, f in r["files"]:
                files = fam.files.setdefault(folder, [])
                if not any(x.name == f.name for x in files):
                    files.append(f)
            # sample with the distilled build's settings when a template has one
            if r["sample"] and (not fam.sample_defaults or (r["distilled"] and not sample_distilled)):
                fam.sample_defaults, sample_distilled = r["sample"], r["distilled"]
            if r["date"] and (fam.added is None or r["date"] < fam.added):
                fam.added = r["date"]
        need = ("checkpoints",) if fam.kind == "checkpoint" else PARTS
        if all(fam.files.get(p) for p in need):  # a download source for every part
            fams[fid] = fam
    return fams


_unsupported: set[str] = set()  # repos whose templates load a shape the harness can't (e.g. checkpoint + separate encoder)


def _template_repos(root: Path, name: str) -> set[str]:
    try:
        w = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    urls: dict[str, dict] = {}
    _model_urls(w, urls)
    return {r.group(1) for m in urls.values() if (r := HF_URL.match(m["url"]))}


def _read_template(root: Path, t: dict, freq: dict[str, int]) -> dict | None:
    """What one template loads, if it's a shape the harness can train: one checkpoint, or one
    diffusion model + one text encoder + one VAE."""
    path = root / f"{t['name']}.json"
    try:
        w = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    by_type: dict[str, list[dict]] = {}
    for n in _nodes(w):
        by_type.setdefault(n.get("type", ""), []).append(n)
    first = lambda nodes: {n["widgets_values"][0] for n in nodes if n.get("widgets_values")}
    unets, clips, vaes, ckpts = (by_type.get(k, []) for k in ("UNETLoader", "CLIPLoader", "VAELoader", "CheckpointLoaderSimple"))
    # a prompt-upsampler LLM loads through CLIPLoader too (Flux 3 Klein): not the text encoder
    clips = [n for n in clips if not (n.get("widgets_values") and NOT_TE.search(str(n["widgets_values"][0])))]
    if ckpts and not unets and not clips and len(first(ckpts)) == 1:
        kind, clip_type = "checkpoint", None
        picks = [("checkpoints", ckpts[0]["widgets_values"][0])]
    elif not ckpts and len(first(unets)) == 1 and len(first(clips)) == 1 and vaes:
        kind, clip_type = "split", clips[0]["widgets_values"][1]
        # an audio-video model loads its audio VAE too: the picture's is the one to encode with
        vae = next((n for n in vaes if not re.search(r"audio", str(n["widgets_values"][0]), re.I)), vaes[0])
        picks = [("diffusion_models", unets[0]["widgets_values"][0]),
                 ("text_encoders", clips[0]["widgets_values"][0]),
                 ("vae", vae["widgets_values"][0])]
    else:
        return None
    urls: dict[str, dict] = {}
    _model_urls(w, urls)
    # a "distilled" template's model is distilled whatever its file is called (flux3-image-klein-*)
    distilled_tmpl = bool(DISTILLED.search(t["name"]))
    files = [(folder, ModelFile(fname, folder, urls[fname]["url"], from_template=True,
                                distilled=bool(DISTILLED.search(fname)) or (distilled_tmpl and folder in ("diffusion_models", "checkpoints"))))
             for folder, fname in picks if fname in urls]
    if len(files) != len(picks):
        return None
    m = HF_URL.match(files[0][1].url)
    sample = None
    ks = by_type.get("KSampler", [])
    if ks and len(ks[0].get("widgets_values", [])) >= 6:
        v = ks[0]["widgets_values"]
        sample = {"model": picks[0][1], "steps": v[2], "cfg": v[3], "sampler": v[4], "scheduler": v[5]}
    return {"template": t["name"], "name": _family_name(t.get("models", []), freq) or t["name"],
            "title": t.get("title") or t["name"], "kind": kind,
            "clip_type": clip_type, "files": files, "repo": m.group(1) if m else None, "sample": sample,
            "date": t.get("date"), "distilled": files[0][1].distilled}


# ---- Hugging Face repo listings -------------------------------------------------------------

def _cache_dir() -> Path:
    import folder_paths

    d = Path(folder_paths.get_user_directory()) / "zentrainer" / "cache" / "hf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def repo_files(repo: str, refresh: bool = False, offline: bool = False) -> list[dict]:
    """[{path, size}] for every file in a Hugging Face model repo; cached on disk for a day.
    `offline` only reads the cache (any age) and raises when there is none."""
    cache = _cache_dir() / f"{repo.replace('/', '__')}.json"
    if cache.is_file() and (offline or (not refresh and time.time() - cache.stat().st_mtime < REPO_TTL)):
        return json.loads(cache.read_text(encoding="utf-8"))
    if offline:
        raise LookupError(f"{repo} not listed yet")
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true", headers=_hf_headers())
    with urllib.request.urlopen(req, timeout=20) as r:
        rows = [{"path": f["path"], "size": f.get("size")} for f in json.load(r) if f.get("type") == "file"]
    cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def _add_repo_files(fam: Family, refresh: bool = False, offline: bool = False) -> None:
    listed = False
    for repo in fam.repos:
        try:
            rows = repo_files(repo, refresh, offline)
        except LookupError:
            continue
        except Exception as e:  # offline, rate-limited, gated
            log.info("[ZenTrainer] can't list %s: %s", repo, e)
            continue
        listed = True
        sizes = {}
        # other precisions of a build a distilled template loads are distilled too
        distilled_variants = {_variant(f.name) for files in fam.files.values() for f in files if f.distilled}
        for row in rows:
            p = row["path"]
            if not p.endswith((".safetensors", ".sft")):
                continue
            parent = p.rsplit("/", 2)[-2] if "/" in p else ""
            folder = FOLDERS.get(parent)
            if folder is None or (fam.kind == "split" and folder not in PARTS) or (fam.kind == "checkpoint" and folder != "checkpoints"):
                continue
            name = p.rsplit("/", 1)[-1]
            if folder == "text_encoders" and NOT_TE.search(name):
                continue
            if folder == "vae" and NOT_PICTURE_VAE.search(name):  # an AV model's audio VAE can't encode images
                continue
            sizes[name] = row.get("size")
            files = fam.files.setdefault(folder, [])
            if not any(f.name == name for f in files):
                files.append(ModelFile(name, folder, f"https://huggingface.co/{repo}/resolve/main/{p}", size=row.get("size"),
                                       distilled=bool(DISTILLED.search(name)) or _variant(name) in distilled_variants))
        for files in fam.files.values():
            for f in files:
                if f.size is None and f.name in sizes:
                    f.size = sizes[f.name]
    fam.repo_listed = fam.repo_listed or (listed and not offline)


# ---- repos without a template ------------------------------------------------------------------

def _list_models(query: str) -> list[dict]:
    req = urllib.request.Request(f"https://huggingface.co/api/models?{query}&limit=1000&expand[]=createdAt&expand[]=gated",
                                 headers=_hf_headers())
    with urllib.request.urlopen(req, timeout=20) as r:
        return [{"id": m["id"], "created": (m.get("createdAt") or "")[:10] or None, "gated": bool(m.get("gated"))}
                for m in json.load(r) if m.get("id")]


def publisher_repos(refresh: bool = False, offline: bool = False) -> list[dict]:
    """[{id, created, gated}] for every Comfy-Org repo and each maker's ComfyUI-layout repos;
    cached like repo listings."""
    cache = _cache_dir() / "publishers.json"
    if cache.is_file() and (offline or (not refresh and time.time() - cache.stat().st_mtime < PUBLISHERS_TTL)):
        return json.loads(cache.read_text(encoding="utf-8"))
    if offline:
        raise LookupError("publishers not listed yet")
    repos = _list_models(f"author={ORG}")
    for maker in MAKERS:
        try:
            repos += [m for m in _list_models(f"author={maker}&search=comfy") if "comfy" in m["id"].lower()]
        except Exception as e:
            log.info("[ZenTrainer] can't list %s: %s", maker, e)
    cache.write_text(json.dumps(repos), encoding="utf-8")
    return repos


def _clip_types() -> list[str]:
    try:
        import nodes

        return list(nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0])
    except Exception:
        return []


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_clip_type(repo_name: str, types: list[str]) -> str | None:
    """The CLIPLoader type a repo's name spells: MOSS-Image -> moss_image, HiDream-I1 -> hidream.
    Exact (separators ignored) first, then the longest type the name starts with."""
    name = _norm(re.sub(r"(?i)[_-]?(comfy(ui)?|repackaged|files|archive)", "", repo_name))
    exact = [t for t in types if _norm(t) == name]
    if exact:
        return exact[0]
    prefix = sorted((t for t in types if len(_norm(t)) >= 4 and name.startswith(_norm(t))), key=lambda t: -len(_norm(t)))
    return prefix[0] if prefix else None


def _pretty(repo_name: str) -> str:
    n = re.sub(r"(?i)[_-]?(comfy(ui)?|repackaged|files|archive)", "", repo_name)
    n = re.sub(r"(?<=\d)[_-](?=\d)", ".", n)  # v1-5 -> v1.5
    words = re.sub(r"[_-]+", " ", n).split()
    return " ".join(w.capitalize() if w.isalpha() and w.islower() else w for w in words)  # flux 3 image klein -> Flux 3 Image Klein


def _family_from_repo(repo: str, rows: list[dict], types: list[str], meta: dict | None = None) -> Family | None:
    by_folder: dict[str, list[ModelFile]] = {}
    for row in rows:
        p = row["path"]
        if not p.endswith((".safetensors", ".sft")):
            continue
        parent = p.rsplit("/", 2)[-2] if "/" in p else "checkpoints"  # a bare file at the root is a checkpoint
        folder = FOLDERS.get(parent)
        if folder is None:
            continue
        name = p.rsplit("/", 1)[-1]
        if folder == "text_encoders" and NOT_TE.search(name):
            continue
        by_folder.setdefault(folder, []).append(ModelFile(
            name, folder, f"https://huggingface.co/{repo}/resolve/main/{p}", size=row.get("size"),
            distilled=bool(DISTILLED.search(name))))
    repo_name = repo.split("/", 1)[1]
    if all(by_folder.get(p) for p in PARTS):
        clip_type = match_clip_type(repo_name, types)
        if clip_type is None or len({_norm(f.name.split(".")[0])[:8] for f in by_folder["text_encoders"]}) > 3:
            return None  # can't tell how ComfyUI loads its text encoder
        kind, files = "split", {p: by_folder[p] for p in PARTS}
    elif by_folder.get("checkpoints") and not by_folder.get("diffusion_models"):
        kind, clip_type, files = "checkpoint", None, {"checkpoints": by_folder["checkpoints"]}
    else:
        return None
    return Family(_slug(_pretty(repo_name)), _pretty(repo_name), kind, clip_type, [repo], [], files,
                  repo_listed=True, source="repo", added=(meta or {}).get("created"), gated=bool((meta or {}).get("gated")))


def _add_repo_families(fams: dict[str, Family], offline: bool) -> None:
    """Publisher repos that no template family covers, from the cached listings (the background
    prefetch fills the cache). Also dates and gates the template families' repos."""
    try:
        metas = {m["id"]: m for m in publisher_repos(offline=offline)}
    except Exception:
        return
    for fam in fams.values():
        fam.gated = fam.gated or any(metas.get(r, {}).get("gated") for r in fam.repos)
    covered = {r for f in fams.values() for r in f.repos}
    covered |= (_non_image_repos() | _unsupported) - covered  # a repo a working family uses stays usable
    types = _clip_types()
    for repo, meta in metas.items():
        if repo in covered or NOT_IMAGE.search(repo.split("/", 1)[1]):
            continue
        try:
            rows = repo_files(repo, offline=offline)
        except Exception:
            continue
        fam = _family_from_repo(repo, rows, types, meta)
        if fam and fam.id not in fams:
            fams[fam.id] = fam


def _non_image_repos() -> set[str]:
    """Repos the video/audio/3D templates load from: not image models."""
    root = _templates_dir()
    out: set[str] = set()
    if root is None or not (root / "index.json").is_file():
        return out
    for cat in json.loads((root / "index.json").read_text(encoding="utf-8")):
        for t in cat.get("templates", []):
            if t.get("mediaType") == "image":
                continue
            out |= _template_repos(root, t["name"])
    return out


# ---- install state ---------------------------------------------------------------------------

def _installed_index() -> dict[str, dict[str, str]]:
    """folder -> {basename: the name ComfyUI lists it by}."""
    import folder_paths

    out: dict[str, dict[str, str]] = {}
    for folder in ("diffusion_models", "text_encoders", "vae", "checkpoints"):
        try:
            names = folder_paths.get_filename_list(folder)
        except Exception:
            names = []
        out[folder] = {n.replace("\\", "/").rsplit("/", 1)[-1]: n for n in names}
    return out


def _mark_installed(fam: Family, index: dict[str, dict[str, str]]) -> None:
    for folder, files in fam.files.items():
        for f in files:
            f.installed = index.get(folder, {}).get(f.name)


def _precision(name: str) -> str:
    m = re.search(r"(bf16|fp16|fp32|fp8\w*|int8\w*|nvfp4|mxfp8|fp4\w*)", name)
    return m.group(1) if m else ""


def _variant(name: str) -> str:
    """A build without its precision: flux3-image-klein-int8-convrot.safetensors -> flux3-image-klein-"""
    m = re.search(r"(bf16|fp16|fp32|fp8|int8|nvfp4|mxfp8|fp4|w4a8)", name)
    return name[:m.start()] if m else name.rsplit(".", 1)[0]


def recommend(fam: Family) -> dict:
    """Files to train on and to sample with. Training wants the base model: an installed base
    build first, then the base build in the template's precision, then whatever the template
    loads. Samples default to the template's own (often turbo) model and its sampler settings."""
    train: dict[str, str | None] = {}
    for folder, files in fam.files.items():
        tmpl = next((f for f in files if f.from_template), files[0] if files else None)
        if folder == PARTS[0] or folder == "checkpoints":
            base = sorted((f for f in files if not f.distilled), key=lambda f: not BASE.search(f.name))
            pick = (next((f for f in base if f.installed), None)
                    or next((f for f in base if tmpl and _precision(f.name) == _precision(tmpl.name)), None)
                    or (base[0] if base else None) or tmpl)
        else:
            pick = next((f for f in files if f.from_template and f.installed), None) or next((f for f in files if f.installed), None) or tmpl
        train[folder] = pick.name if pick else None
    model_folder = "checkpoints" if fam.kind == "checkpoint" else PARTS[0]
    sample = None
    sd = fam.sample_defaults
    if sd and sd.get("model") and sd["model"] != train.get(model_folder):
        sample = sd
    return {"train": train, "sample": sample}


def missing(fam: Family, picks: dict[str, str | None]) -> list[dict]:
    out = []
    for folder, name in picks.items():
        f = next((x for x in fam.files.get(folder, []) if x.name == name), None)
        if f and not f.installed:
            out.append({"folder": folder, "name": f.name, "size": f.size})
    return out


# ---- public API --------------------------------------------------------------------------------

_lock = threading.Lock()
_families: dict[str, Family] | None = None


def _all() -> dict[str, Family]:
    global _families
    with _lock:
        if _families is None:
            _families = _from_templates()
            _add_repo_families(_families, offline=True)  # whatever the cache already knows
            _correct(_families)
        return _families


_prefetched = False


def _prefetch() -> None:
    """List every family's repo, and the publishers' other repos, once in the background, so the
    next list knows the base builds, their sizes and the families with no template."""
    repos = [r for fam in list(_all().values()) for r in fam.repos]
    try:
        repos += [m["id"] for m in publisher_repos() if m["id"] not in repos and not NOT_IMAGE.search(m["id"].split("/", 1)[1])]
    except Exception:
        pass
    for repo in repos:
        try:
            repo_files(repo)
        except Exception:
            pass
    with _lock:
        _add_repo_families(_families or {}, offline=True)
        _correct(_families or {})


def relist() -> dict:
    """Re-list the publishers' repos on Hugging Face now and rebuild the catalog, so a model added
    since the last listing shows up without waiting for the cache. The repos' file lists are
    fetched only for repos not listed before (each family's own refresh refetches those).
    Returns the families that are new."""
    global _families
    before = set(_all())
    repos = publisher_repos(refresh=True)  # raises when Hugging Face can't be reached
    for m in repos:
        if NOT_IMAGE.search(m["id"].split("/", 1)[1]):
            continue
        try:
            repo_files(m["id"])  # from the cache when it's there
        except Exception as e:
            log.info("[ZenTrainer] can't list %s: %s", m["id"], e)
    with _lock:
        _families = None
    fams = _all()
    new = sorted(fams[f].name for f in set(fams) - before)
    return {"families": len(fams), "new": new}


def list_families() -> list[dict]:
    global _prefetched
    if not _prefetched:
        _prefetched = True
        threading.Thread(target=_prefetch, name="zentrainer-hf-prefetch", daemon=True).start()
    index = _installed_index()
    out = []
    for fam in sorted(_all().values(), key=lambda f: (f.added or "", f.name.lower()), reverse=True):  # newest first
        if not fam.repo_listed:
            _add_repo_files(fam, offline=True)  # builds already listed once, without a request
        _mark_installed(fam, index)
        out.append(fam.public())
    return out


def get(fid: str, with_repo: bool = False, refresh: bool = False) -> dict | None:
    fam = _all().get(fid)
    if fam is None:
        return None
    if with_repo and (refresh or not fam.repo_listed):
        _add_repo_files(fam, refresh)
    _mark_installed(fam, _installed_index())
    return fam.public()


def find_file(fid: str, folder: str, name: str) -> ModelFile | None:
    """A file this catalog knows (and so may download): the only way a URL reaches the downloader."""
    fam = _all().get(fid)
    if fam is None:
        return None
    return next((f for f in fam.files.get(folder, []) if f.name == name), None)
