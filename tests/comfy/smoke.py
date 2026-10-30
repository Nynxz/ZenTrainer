"""A few real training steps of one model family, on CPU, through a real ComfyUI.

    python tests/comfy/smoke.py <family> <comfyui dir> <work dir>

The model is ComfyUI's own class for the family at a tiny size with random weights, loaded the
way a file is (comfy.sd detects the family from its keys); the text encoder and VAE are stand-ins
of the right widths. Everything else is ZenTrainer's: the trainer, ComfyUI's patches in
harness/compat/, the LoRA, sampling, saving and resuming: what a change in ComfyUI's model code
breaks, which the unit tests' stand-in ComfyUI modules can't show.

Every ComfyUI folder points into the work dir, and CUDA is hidden: a GPU busy training is left
alone. Prints one line of JSON: what test_training.py checks.
"""

import json
import logging
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
FAMILY, COMFYUI, WORK = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sys.path[:0] = [str(COMFYUI), str(Path(__file__).resolve().parents[2] / "engine")]

from comfy.cli_args import args  # noqa: E402

args.cpu = True
import folder_paths  # noqa: E402

for d in ("input", "output", "user", "temp"):
    (WORK / d).mkdir(parents=True, exist_ok=True)
folder_paths.set_input_directory(str(WORK / "input"))
folder_paths.set_output_directory(str(WORK / "output"))
folder_paths.set_user_directory(str(WORK / "user"))
folder_paths.set_temp_directory(str(WORK / "temp"))

import torch  # noqa: E402


def moss():
    from comfy.ldm.moss_image.model import MOSSImage

    # detection: dim from the modulation, one head per 128 dims; text 4096 wide (not detected)
    return lambda ops: MOSSImage(dim=128, ffn_dim=256, num_heads=1, num_layers=2, operations=ops), 4096, (128, 16)


def krea2():
    from comfy.ldm.krea2.model import SingleStreamDiT

    # detection: 128-dim heads; the text side keeps its 20 heads (not detected), so 20 * 8 wide
    return (lambda ops: SingleStreamDiT(features=256, heads=2, kvheads=1, layers=2, channels=16, txtlayers=2,
                                        txtdim=160, operations=ops), 2 * 160, (16, 8))


def fibo2():
    from comfy.ldm.fibo2.model import Fibo2

    # detection: dim and gist tokens (each 256 at least) from the perceiver's queries, one head per
    # 128 dims. The text is 7680 wide (not detected) for the perceiver and again per injection layer
    return (lambda ops: Fibo2(dim=256, n_heads=2, n_layers=2, n_refiner_layers=1, injection_layers=(1,),
                              perceiver_layers=1, max_gist_tokens=256, ffn_dim=256, operations=ops),
            2 * 7680, (128, 16))


def flux3():
    from comfy.ldm.flux3.model import Flux3

    # detection: one head per 128 dims; text 20480 wide (fixed). Two of each block: the lists
    # gradient checkpointing wraps, as in the real model
    return (lambda ops: Flux3(hidden_size=128, num_heads=1, depth=2, depth_single_blocks=2, operations=ops), 20480,
            (128, 16))


def h3():
    from comfy.ldm.minimax.model import MiniMaxH3Model

    # video and sound in one model: pictures train as one-frame clips beside silence
    return (lambda ops: MiniMaxH3Model(hidden_size=256, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                                       ffn_hidden_size=256, text_dim=64, time_embed_hidden_size=256, time_embed_dim=128,
                                       operations=ops), 64, (24, 16))


def anima():
    from comfy.ldm.anima.model import Anima

    # Cosmos Predict2 with a text adapter. detection: model_channels from the patch embedder, the
    # blocks counted, 16 heads at a width it doesn't know, the rest fixed (rope3d, AdaLN-LoRA 256,
    # 1024-wide text); its adapter is full size. Wan 2.1's latents: a picture is a one-frame clip
    return (lambda ops: Anima(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
                              patch_spatial=2, patch_temporal=1, model_channels=512, num_blocks=2, num_heads=16,
                              crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
                              use_adaln_lora=True, adaln_lora_dim=256, extra_per_block_abs_pos_emb=False,
                              rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
                              rope_t_extrapolation_ratio=1.0, image_model="anima", operations=ops),
            1024, (16, 8))


# each: (the model at a tiny size, from ComfyUI's operations), its text width, its VAE's (latent
# channels, downscale)
FAMILIES = {"moss": moss, "krea2": krea2, "fibo2": fibo2, "flux3": flux3, "h3": h3, "anima": anima}


class Clip:
    """The text encoder: a fixed random vector per word, as wide as the model reads."""

    def __init__(self, width):
        self.width = width

    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, text):
        words = text.split() or [""]
        rows = [torch.randn(self.width, generator=torch.Generator().manual_seed(sum(map(ord, w)))) for w in words]
        return [[torch.stack(rows)[None], {}]]


class Vae:
    """Pictures to latents and back: average pooling and a fixed projection to the latent channels.
    A video VAE's latents (the model's latent format has 3 dimensions) carry a frame axis, as the
    real one's do: (B, C, 1, h, w) for a picture."""

    def __init__(self, latent_format, channels, down):
        self.proj = torch.randn(channels, 3, generator=torch.Generator().manual_seed(1))
        self.down = down
        self.video = getattr(latent_format, "latent_dimensions", 2) == 3

    def spacial_compression_encode(self):
        return self.down

    def encode(self, pixels):  # (B, H, W, 3)
        x = torch.nn.functional.avg_pool2d(pixels.movedim(-1, 1).float(), self.down)
        x = torch.einsum("cd,bdhw->bchw", self.proj, x)
        return x.unsqueeze(2) if self.video else x

    def decode(self, latent):  # -> (B, H, W, 3), or (B, T, H, W, 3) from a video VAE
        frames = latent.movedim(2, 1).flatten(0, 1) if latent.ndim == 5 else latent
        x = torch.einsum("cd,bchw->bdhw", self.proj, frames.float()).sigmoid()
        x = torch.nn.functional.interpolate(x, scale_factor=self.down).movedim(1, -1)
        return x.unflatten(0, latent.shape[:1] + latent.shape[2:3]) if latent.ndim == 5 else x


def tiny_model(build):
    """The family's ComfyUI model, loaded from a state dict as a file of it would be."""
    import comfy.ops
    import comfy.sd

    torch.manual_seed(0)
    m = build(comfy.ops.disable_weight_init)
    with torch.no_grad():
        for p in m.parameters():  # disable_weight_init leaves them uninitialised
            p.normal_(0, 0.1)
        for b in m.buffers():  # rope frequencies and the like: (0, 1]
            b.uniform_(0.01, 1)
    return comfy.sd.load_diffusion_model_state_dict(m.state_dict(), model_options={})


def dataset(root: Path):
    from PIL import Image

    root.mkdir(parents=True)
    for i, (cap, rgb) in enumerate([("a red square", (220, 40, 40)), ("a blue circle on grey", (40, 40, 220))]):
        Image.new("RGB", (64, 64), rgb).save(root / f"{i}.png")
        (root / f"{i}.txt").write_text(cap)


def stock_load(patcher, path: str) -> tuple[int, list[str]]:
    """The saved LoRA through stock ComfyUI (its LoraLoader's path): (weights it patches, keys it
    leaves unloaded)."""
    import comfy.sd
    import comfy.utils

    missing = []

    class Catch(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if msg.startswith("lora key not loaded"):
                missing.append(msg.split(": ", 1)[1])

    h = Catch()
    logging.getLogger().addHandler(h)
    try:
        loaded, _ = comfy.sd.load_lora_for_models(patcher, None, comfy.utils.load_torch_file(path), 1.0, 0)
    finally:
        logging.getLogger().removeHandler(h)
    return len(loaded.patches), missing


def patch_drift(spec) -> float:
    """How far (relative to its size) ZenTrainer's patched model (harness/compat/, gradients on) is
    from ComfyUI's own code (in training mode, without gradients, where it runs as written) on one
    picture: a patch is a copy of ComfyUI's code, and this is what says when ComfyUI's changed
    under it."""
    import comfy.model_management as mm

    from zentrainer.harness import compat, trainer

    apply = compat.apply
    compat.apply = lambda qk_rope=False: None
    t = trainer.Trainer(spec, "drift", emit=lambda kind, data: None, callbacks=[])
    try:
        with torch.inference_mode(False):
            t.setup()
            compat.apply = apply
            t.enter_training()
            x0, _, shapes = t.batch([0])
            conds, sigma = [t.fixed_cond(0, x0, shapes)], torch.full((1,), 0.5)
            with torch.no_grad():
                want = t.denoiser(x0, sigma, conds)
            apply(qk_rope=spec.memory.fused_qk_rope)
            got = t.denoiser(x0.clone().requires_grad_(True), sigma, conds)
    finally:
        compat.apply = apply
        mm.in_training = False
        t.network.eject(t.models.patcher)
        t._release()
    parts = list(zip(got.unbind(), want.unbind()) if got.is_nested else [(got, want)])  # H3: video and sound
    return max((a.detach() - b).abs().max().item() / b.abs().max().item() for a, b in parts)


def main():
    import safetensors.torch

    from zentrainer import paths
    from zentrainer.harness import models, trainer
    from zentrainer.runs.store import RunStore
    from zentrainer.spec import TrainSpec

    build, text_width, (channels, down) = FAMILIES[FAMILY]()
    patcher = tiny_model(build)
    vae = Vae(patcher.model.latent_format, channels, down)
    models.load = lambda spec: models.Models(patcher, Clip(text_width), vae)
    dataset(WORK / "input" / "smoke")
    spec = {
        "name": "smoke",
        "model": {"kind": "split", "unet_name": "tiny", "clip_name": "tiny", "vae_name": "tiny", "clip_type": "tiny"},
        "dataset": {"folder": "smoke", "resolution": 64, "buckets": False},
        "optim": {"steps": 4},
        "memory": {"training_dtype": "fp32"},
        # validation's own interval misses the saves: keeping the best by it puts it on each one
        "validation": {"every": 100, "images": 1, "noise_levels": 2},
        "output": {"save_every": 1, "keep": "val_loss", "keep_count": 1},
        "samples": {"every": 4, "prompts": [{"prompt": "a red square"}], "width": 64, "height": 64, "steps": 2,
                    "baseline": False, "likeness": False},
    }
    drift = patch_drift(TrainSpec.from_dict(spec))
    events = []
    store = RunStore(WORK / "runs")
    store.create("smoke1", spec)
    t = trainer.Trainer(TrainSpec.from_dict(spec), "smoke1", store=store,
                        emit=lambda kind, data: events.append((kind, data)))
    status = t.run()
    saves = [d for k, d in events if k == trainer.events.CHECKPOINT]
    final = next(d["path"] for d in saves if d["final"])
    lora = safetensors.torch.load_file(final)
    out = {
        "family": type(patcher.model).__name__, "drift": drift, "status": status, "loss": t.last_loss, "steps": t.step,
        "lora_keys": len(lora), "trained": any(v.abs().max().item() > 0 for k, v in lora.items() if "lora_up" in k),
        **dict(zip(("patched", "unloaded"), stock_load(patcher, final))),
        "samples": [str(paths.output_root() / "samples" / "smoke1" / d["filename"])
                    for k, d in events if k == trainer.events.SAMPLE],
    }
    kept = store.read("smoke1", "checkpoints")
    out.update(validated=sorted({r["step"] for r in store.read("smoke1", "validation")}),
               pruned=[r["step"] for r in store.read("smoke1", "pruned")],
               pruned_files=[Path(r["path"]).exists() for r in store.read("smoke1", "pruned")],
               kept=[c["step"] for c in kept if not c["final"]],
               kept_files=all(Path(c["path"]).is_file() for c in kept),
               best=min(range(1, 4), key=lambda s: next(r["val_loss"] for r in store.read("smoke1", "validation")
                                                          if r["step"] == s)))
    # resume from the last checkpoint and its state, on to step 6
    last = saves[-1]
    t2 = trainer.Trainer(TrainSpec.from_dict({**spec, "optim": {"steps": 6}, "samples": {"every": 0}}), "smoke1",
                         emit=lambda kind, data: None, resume_from=last["path"], resume_state=last["state"])
    out.update(resumed=t2.run(), resumed_steps=t2.step, resumed_loss=t2.last_loss)
    print(json.dumps(out))


main()
