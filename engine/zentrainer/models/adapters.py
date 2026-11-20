"""LoRAs a run trains on top of (harness/network.py load_training_adapter), two kinds:

- training adapters, on while training only. Distilled models (turbo, guidance-distilled) train
  badly on their own: the LoRA spends itself undoing the distillation and samples drift. A
  de-distillation adapter under it absorbs that, and is left out of samples and the saved file.
- base LoRAs, on always (training, samples, use): a fix the model always needs, or an edit LoRA a
  specialist builds on. The trained LoRA learns only what's left; it's saved without the base and
  records it (network.base_lora).

This module is the pickers' catalog:

- curated: adapters we know, per model family, with their Hugging Face file;
- custom: ones the user added from a Hugging Face link (kept in <user>/zentrainer/adapters.json);
- local: anything already in a loras folder's training_adapters/ subfolder.

Downloads land in <loras>/training_adapters/ (or base_loras/), beside any already installed. Only
catalog entries download: the user adds a link once (validated against the repo's file list), then picks it.
"""

from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import paths, safeio
from . import families

SUBDIR = "training_adapters"
SUBDIRS = {"training": SUBDIR, "base": "base_loras"}
HF = "https://huggingface.co"


@dataclass
class Adapter:
    id: str
    label: str
    repo: str
    file: str  # path in the repo
    families: list[str] = field(default_factory=list)  # family ids it suits; empty = any
    note: str = ""
    save_as: str = ""  # local file name, when the repo's is generic (pytorch_lora_weights)
    size: int | None = None
    source: str = "curated"  # curated | custom
    suggest: bool = False  # offered when the family is picked and no adapter is set
    kind: str = "training"  # training (on while training only) | base (on always)

    @property
    def name(self) -> str:
        return self.save_as or Path(self.file).name

    @property
    def url(self) -> str:
        return f"{HF}/{self.repo}/resolve/main/{self.file}"


CURATED = [
    Adapter("circlestone-h3-image", "circlestone image adapter", "circlestone-labs/MiniMax-H3-Image-Training-Adapter",
            "minimax_h3_image_training_adapter.safetensors", ["minimax-h3"],
            "De-distills H3 and leans it toward stills. Best for image datasets.", size=620284872, suggest=True),
    Adapter("ostris-h3-image-v1", "Ostris H3-Image adapter", "ostris/minimax_h3_training_adapter",
            "minimax_h3_image_training_adapter_v1.safetensors", ["minimax-h3-image"],
            "De-distills MiniMax-H3-Image for training.", size=155110072, suggest=True),
    Adapter("ostris-h3-v3", "Ostris adapter v3", "ostris/minimax_h3_training_adapter",
            "minimax_h3_training_adapter_v3.safetensors", ["minimax-h3"],
            "Ostris's general H3 adapter (ai-toolkit). Video and image.", size=310168800),
    Adapter("ostris-h3-ref2va-v3", "Ostris ref-to-video adapter v3", "ostris/minimax_h3_training_adapter",
            "minimax_h3_ref2va_training_adapter_v3.safetensors", ["minimax-h3"],
            "For the reference-to-video build.", size=310168552),
    Adapter("ostris-fasth3-8step", "Ostris FastH3 8-step adapter", "ostris/minimax_h3_training_adapter",
            "fastvideo_fasth3_8step_v2_training_adapter_v1.safetensors", ["minimax-h3"],
            "For FastVideo's 8-step H3 build.", size=350323144),
    Adapter("ostris-krea2-turbo", "Ostris Krea 2 Turbo adapter", "ostris/krea2_turbo_training_adapter",
            "krea2_turbo_training_adapter_v1.safetensors", ["krea-2"],
            "Train on Krea 2 Turbo without breaking its few-step sampling.", size=228587504),
    Adapter("nynxz-nk2e-v03", "NK2E v0.3 edit base (Krea 2)", "nynxz/NK2E", "comfy/v0.3/NK2E-v0.3.safetensors",
            ["krea-2"], "Nynxz's in-context edit LoRA: train a specialist edit (Edit training on) on top of it, "
            "then use both together.", size=385418784, kind="base"),
    Adapter("nynxz-moss-sft-destripe", "MOSS SFT destripe", "nynxz/MOSS-Image",
            "loras/moss_sft_destripe_top64_bf16.safetensors", ["moss-image"],
            "Removes the SFT build's banding. Under a LoRA, the LoRA doesn't spend itself learning that fix.",
            size=1900586, kind="base"),
    Adapter("ostris-zimage-turbo-v2", "Ostris Z-Image Turbo adapter v2", "ostris/zimage_turbo_training_adapter",
            "zimage_turbo_training_adapter_v2.safetensors", ["z-image-turbo"],
            "Train on Z-Image Turbo without breaking its few-step sampling.", size=340194488),
    Adapter("ostris-flux1-schnell", "Ostris FLUX.1 schnell adapter", "ostris/FLUX.1-schnell-training-adapter",
            "pytorch_lora_weights.safetensors", ["flux-1"], "For training on FLUX.1 schnell.",
            save_as="flux1_schnell_training_adapter.safetensors", size=451202808),
    Adapter("ostris-qwen-image-2-turbo", "Ostris Qwen-Image 2 Turbo adapter", "ostris/qwen_image_2_turbo_training_adapter",
            "qwen_image_2_turbo_training_adapter_v1.safetensors", ["qwen-image"], "For Qwen-Image 2 Turbo.", size=159436160),
    Adapter("kigland-klein-4b", "Flux.2 Klein 4B adapter (community)", "kigland/flux2_klein_4b_training_adapter",
            "flux2_klein_4b_training_adapter_v1.safetensors", ["flux-2-klein"], "Community adapter for the distilled Klein 4B.",
            size=96365704),
]

_lock = threading.Lock()


def _store() -> Path:
    import folder_paths

    return Path(folder_paths.get_user_directory()) / "zentrainer" / "adapters.json"


def _custom() -> list[Adapter]:
    p = _store()
    if not p.is_file():
        return []
    try:
        return [Adapter(**{**row, "source": "custom"}) for row in safeio.read_json(p, [])]
    except TypeError:  # rows from another version: keep the file for the user, start a fresh list
        safeio.set_aside(p)
        return []


def _save_custom(rows: list[Adapter]) -> None:
    p = _store()
    p.parent.mkdir(parents=True, exist_ok=True)
    safeio.write_json(p, [asdict(a) for a in rows])


def catalog() -> list[Adapter]:
    return CURATED + _custom()


def _installed() -> dict[str, str]:
    """basename -> the name ComfyUI lists it by, for every LoRA file."""
    import folder_paths

    return {Path(n).name: n for n in folder_paths.get_filename_list("loras")}


def listing(family: str | None, kind: str = "training") -> list[dict]:
    """Catalog entries of this kind (the family's first), then local files the catalog doesn't
    name: training adapters from a training_adapters/ folder, base LoRAs from any loras folder."""
    import folder_paths

    have = _installed()
    out, named = [], set()
    for a in catalog():
        if a.kind != kind:
            continue
        listed = have.get(a.name)
        named.add(a.name)
        fits = not a.families or (family in a.families if family else False)
        out.append({**asdict(a), "name": a.name, "installed": listed, "fits": fits, "url": f"{HF}/{a.repo}"})
    for n in folder_paths.get_filename_list("loras"):
        in_sub = n.replace("\\", "/").startswith(SUBDIR + "/")
        if (in_sub if kind == "training" else not in_sub) and Path(n).name not in named:
            out.append({"id": f"local:{n}", "label": Path(n).stem, "name": Path(n).name, "installed": n, "source": "local",
                        "families": [], "fits": True, "note": "", "repo": "", "file": "", "size": None, "url": "",
                        "suggest": False, "save_as": ""})
    out.sort(key=lambda r: (not r["fits"], r["source"] == "local"))
    return out


def target_dir(kind: str = "training") -> Path:
    """Where adapters download: a loras folder that already has the kind's subfolder, else the
    loras folder downloads use (not ZenTrainer's own output folder of trained LoRAs)."""
    import folder_paths

    from . import downloads

    outs = {Path(folder_paths.get_output_directory()).resolve(), *(r.resolve() for r in paths.output_roots())}
    roots = [Path(p) for p in folder_paths.get_folder_paths("loras") if not outs & set(Path(p).resolve().parents)]
    sub = SUBDIRS.get(kind, SUBDIR)
    for r in roots:
        if (r / sub).is_dir():
            return r / sub
    return downloads.target_dir("loras") / sub


def download(adapter_id: str):
    from . import downloads

    a = next((a for a in catalog() if a.id == adapter_id), None)
    if a is None:
        raise ValueError(f"no adapter {adapter_id!r}")
    return downloads.manager().enqueue("adapter", "loras", a.name, a.url, target_dir(a.kind), a.size)


_HF_LINK = re.compile(r"^(?:https?://huggingface\.co/)?([\w.-]+/[\w.-]+)(?:/(?:blob|resolve|tree)/[\w.-]+/?(.*))?/?$")


def add(link: str, family: str | None, file: str | None = None, kind: str = "training") -> dict:
    """Add an adapter from a Hugging Face link (repo, or a file in it). A repo with several
    .safetensors files and no file chosen returns {choices} to pick from instead."""
    m = _HF_LINK.match(link.strip())
    if not m:
        raise ValueError("paste a Hugging Face link: https://huggingface.co/owner/repo (or a file in it)")
    repo, path = m.group(1), (file or m.group(2) or "").strip("/")
    try:
        rows = [r for r in families.repo_files(repo, refresh=True) if r["path"].endswith(".safetensors")]
    except Exception as e:
        raise ValueError(f"couldn't list {repo}: {e}") from e
    if path:
        rows = [r for r in rows if r["path"] == path]
        if not rows:
            raise ValueError(f"{path} isn't a .safetensors file in {repo}")
    if not rows:
        raise ValueError(f"{repo} has no .safetensors files")
    if len(rows) > 1:
        return {"choices": [{"file": r["path"], "size": r.get("size")} for r in rows], "repo": repo}
    row = rows[0]
    base = Path(row["path"]).name
    owner = repo.split("/")[0]
    save_as = f"{owner}_{repo.split('/')[1]}.safetensors" if base == "pytorch_lora_weights.safetensors" else ""
    a = Adapter(f"custom{'' if kind == 'training' else '-' + kind}:{repo}/{row['path']}", f"{Path(base).stem} ({owner})", repo, row["path"],
                [family] if family else [], "", save_as, row.get("size"), "custom", kind=kind)
    with _lock:
        rows_ = [c for c in _custom() if c.id != a.id] + [a]
        _save_custom(rows_)
    return {"adapter": {**asdict(a), "name": a.name}}


def remove(adapter_id: str) -> bool:
    """Forget a custom adapter (the downloaded file stays)."""
    with _lock:
        rows = _custom()
        kept = [c for c in rows if c.id != adapter_id]
        if len(kept) == len(rows):
            return False
        _save_custom(kept)
    return True
