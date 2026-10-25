"""Trainable adapters (LoRA, LoHa, LoKr, OFT) on a ComfyUI ModelPatcher.

Built on `comfy.weight_adapter`, so trained files load with ComfyUI's own LoRA loader.
Adapters attach as ModelPatcher weight wrappers (applied when each weight is cast to
the compute device, so lowvram/partial loading keeps working) or, in bypass mode, as
forward hooks for quantized bases whose weights can't be patched.

Adapters are pinned to the training device: ComfyUI may offload base weights, but
must not move the parameters the optimizer holds state for.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re

import torch

from ..spec import NetworkSpec

# Target presets match on structure, not per-family layer names: a linear/conv layer
# counts as attention (or MLP) when any *enclosing* module's name says so. That covers
# SD's attn1.to_q / ff.net.0.proj, Flux's img_attn.qkv / img_mlp.0, Krea 2's attn.wq /
# mlp.up, Llama-style self_attn.q_proj / mlp.gate_proj, ... without listing each.
_ATTN = re.compile(r"attn|attention")
_MLP = re.compile(r"^(ff|ffn|ff_context|feed_forward|feedforward)$|mlp")
# fused attention+MLP projections that sit directly in a block (Flux single blocks)
_FUSED = re.compile(r"^(linear1|linear2)$")
# some models only say it in the layer's own name (MOSS Image: layers.3.proj.q, .proj.ffn_up_gate,
# text_embedder.blocks.0.ffn_up). Trusted only inside a numbered block, so embedders and
# modulation layers at the top level stay out.
_ATTN_LEAF = re.compile(r"^(q|k|v|o|qkv|wq|wk|wv|wo|to_[qkvo]\w*|[qkvo]_proj|qkv_proj|out_proj)$")
_MLP_LEAF = re.compile(r"^(ffn_\w+|ff_\w+|fc[12]|w[123]|gate_proj|up_proj|down_proj|mlp_\w+)$")
_BLOCK = re.compile(r"^\d+$")
# layers an adapter would never train: text-side ones that run once per caption outside the
# denoising graph (MiniMax H3's token refiner, under no_grad when conds are prepared), and
# H3's sparse-attention gate, which the dense forward never calls
_PREPARED = re.compile(r"(^|\.)(token_refiner|condition_proj|to_gate_compress)(\.|$)")


_BLOCK_INDEX = re.compile(r"(?:^|\.)(?:blocks|double_blocks|single_blocks|joint_blocks|transformer_blocks|layers)\.(\d+)(?:\.|$)")


def in_blocks(name: str, blocks: str) -> bool:
    """Is the layer inside the block range ("20-49", "20-", "-10", "7")? Layers outside any
    numbered block (embedders, final layers) are out when a range is set."""
    if not blocks.strip():
        return True
    m = _BLOCK_INDEX.search(name)
    if m is None:
        return False
    i = int(m.group(1))
    lo, _, hi = blocks.replace(" ", "").partition("-")
    if not _:  # a single block
        return i == int(lo)
    return (not lo or i >= int(lo)) and (not hi or i <= int(hi))


def skipped(name: str, skip: str) -> bool:
    """Is the layer in one of the skipped blocks ("22", "22, 30-31")?"""
    if not skip.strip():
        return False
    m = _BLOCK_INDEX.search(name)
    if m is None:
        return False
    i = int(m.group(1))
    for part in skip.replace(" ", "").split(","):
        lo, dash, hi = part.partition("-")
        if (int(lo) <= i <= int(hi)) if dash else i == int(lo):
            return True
    return False


def matches(name: str, targets: str | list[str]) -> bool:
    """Should the layer at module path `name` get an adapter?"""
    if not isinstance(targets, str):
        return any(re.search(p, name) for p in targets)
    if targets == "all":
        return True
    if _PREPARED.search(name):
        return False
    *parents, leaf = name.split(".")
    in_block = any(_BLOCK.match(p) for p in parents)
    in_attn = any(_ATTN.search(p) for p in parents) or (in_block and bool(_ATTN_LEAF.match(leaf)))
    if targets == "attn":
        return in_attn
    in_mlp = any(_MLP.search(p) for p in parents) or (in_block and bool(_MLP_LEAF.match(leaf)))
    if targets == "mlp":  # a fused attention+MLP projection (Flux's linear1/2) only goes with attn_mlp
        return in_mlp
    return in_attn or in_mlp or bool(_FUSED.match(leaf))  # attn_mlp


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class _Pinned:
    """Weight-wrapper shim: applies the adapter, ignores ComfyUI's device moves."""

    def __init__(self, adapter: torch.nn.Module):
        self.adapter = adapter

    def __call__(self, w):
        return self.adapter(w)

    def passive_memory_usage(self) -> int:
        return sum(p.nelement() * p.element_size() for p in self.adapter.parameters())

    def move_to(self, device=None) -> int:
        return self.passive_memory_usage()


class _BiasDiff(torch.nn.Module):
    """Additive delta for 1-D weights (norms) and biases; scaled by the switch's strength when it
    has one (a scalable network)."""

    def __init__(self, shape, dtype, switch=None):
        super().__init__()
        self.diff = torch.nn.Parameter(torch.zeros(shape, dtype=dtype))
        self.switch = switch

    def forward(self, b):
        st = self.switch.strength if self.switch is not None else 1.0
        if st == 0.0:
            return b
        return (b.to(self.diff) + (self.diff if st == 1.0 else st * self.diff)).to(b.dtype)

    __call__ = forward


class _Assisted(torch.nn.Module):
    """A layer's trainable adapter (or none) plus a frozen training adapter that's only on
    while training (`switch.on`): e.g. a de-distillation adapter for a guidance-distilled
    model, which training must sit on top of but samples and the saved LoRA must not include.
    A base LoRA (`base`) is on always, training and samples: the trained one sits on it.
    Speaks both attachment protocols: called on the weight (weight wrapper) and h/g (bypass)."""

    _BYPASS_ATTRS = ("multiplier", "is_conv", "conv_dim", "kernel_size", "in_channels", "out_channels", "kw_dict")

    def __init__(self, trained, assist, switch, preview=None, preview_strength: float = 1.0,
                 base=None, base_strength: float = 1.0):
        super().__init__()
        self.trained = trained
        self.base = base  # on always
        self.base_strength = base_strength
        self.assist = assist  # on while training (switch.on)
        self.preview = preview  # on while sampling: a turbo LoRA for few-step previews
        self.preview_strength = preview_strength
        self.switch = switch

    def __call__(self, w):
        if self.base is not None:
            w = w + self.base_strength * (self.base(w) - w)
        if self.switch.on and self.assist is not None:
            w = self.assist(w)
        elif not self.switch.on and self.preview is not None:
            w = w + self.preview_strength * (self.preview(w) - w)
        st = self.switch.strength
        if self.trained is None or st == 0.0:
            return w
        return self.trained(w) if st == 1.0 else w + st * (self.trained(w) - w)

    def _share(self) -> None:  # bypass sets layer info on the adapter it holds: pass it on
        for child in (self.trained, self.assist, self.preview, self.base):
            if child is not None:
                for a in self._BYPASS_ATTRS:
                    if hasattr(self, a):
                        setattr(child, a, getattr(self, a))

    def h(self, x, base_out):
        self._share()
        st = self.switch.strength
        out = 0
        if self.trained is not None and st != 0.0:
            out = self.trained.h(x, base_out)
            if st != 1.0:
                out = st * out
        if self.base is not None:
            out = out + self.base_strength * self.base.h(x, base_out)
        if self.switch.on and self.assist is not None:
            out = out + self.assist.h(x, base_out)
        elif not self.switch.on and self.preview is not None:
            out = out + self.preview_strength * self.preview.h(x, base_out)
        return out

    def g(self, y):
        return self.trained.g(y) if self.trained is not None else y


class _Switch:
    """Shared by every layer of a network: `on` = training (assist adapters on, previews off);
    `strength` = how much of the trained adapter applies (a slider trains at +1 and -1, and its
    teacher pass is the base model at 0). Only layers wrapped in _Assisted see it."""

    on = True
    strength = 1.0


class _Sliced(torch.nn.Module):
    """Frozen adapters on row slices of one fused weight: a LoRA made for separate q/k/v on a
    build that fuses them (MOSS-Image int8's proj.qkv). ComfyUI maps those keys to
    (weight, (dim 0, offset, size)); each adapter acts on its slice. Both protocols, like
    _Assisted: called on the weight, and h/g (bypass) on the output's last dim."""

    _BYPASS_ATTRS = _Assisted._BYPASS_ATTRS

    def __init__(self):
        super().__init__()
        self.parts: list[tuple[int, int, torch.nn.Module]] = []

    def add(self, offset: int, size: int, adapter) -> None:
        self.parts.append((offset, size, adapter))
        self.add_module(f"part{len(self.parts)}", adapter)

    def __call__(self, w):
        out = w.clone()
        for off, size, a in self.parts:
            out[off:off + size] = a(w[off:off + size])
        return out

    def h(self, x, base_out):
        out = torch.zeros_like(base_out)
        for off, size, a in self.parts:
            for k in self._BYPASS_ATTRS:
                if hasattr(self, k):
                    setattr(a, k, getattr(self, k))
            out[..., off:off + size] = a.h(x, base_out[..., off:off + size])
        return out

    def g(self, y):
        return y


def load_training_adapter(patcher, path: str, device, dtype, skipped: list | None = None) -> dict:
    """A LoRA file as frozen train-form adapters keyed by model weight ("diffusion_model.X.weight"),
    read with ComfyUI's own LoRA key mapping (diffusers, PEFT lora_A/B, kohya names ...)."""
    import comfy.lora
    import comfy.utils
    from comfy.weight_adapter.lora import LoRAAdapter, LoraDiff

    sd = comfy.utils.load_torch_file(path, safe_load=True)
    patches = comfy.lora.load_lora(sd, comfy.lora.model_lora_keys_unet(patcher.model, {}), log_missing=False)
    out = {}
    left_out = []
    for key, adapter in patches.items():
        sliced = None
        if isinstance(key, tuple):  # (weight, (dim, offset, size)): a slice of a fused weight
            key, (dim, offset, size) = key[0], key[1][:3]
            sliced = (offset, size) if dim == 0 else None
            if sliced is None:
                left_out.append(key)
                continue
        if isinstance(adapter, LoRAAdapter):
            mat1, mat2, alpha, mid, dora, reshape = adapter.weights
            fits = (_fits_slice(patcher.model, key, sliced, (mat1.shape[0], mat2[0].numel())) if sliced
                    else _fits(patcher.model, key, (mat1.shape[0], mat2[0].numel())))
            if not fits:
                left_out.append(key)  # same name, different layer (another build of the model): ComfyUI skips these too
                continue
            alpha = float(alpha) if alpha is not None else float(mat2.shape[0])  # no alpha: scale 1
            module = LoraDiff((mat1, mat2, alpha, mid, dora, reshape))
        elif hasattr(adapter, "to_train"):
            module = adapter.to_train()
        else:
            continue
        module = module.to(device=device, dtype=dtype).requires_grad_(False)
        if sliced:
            if not isinstance(out.get(key), _Sliced):
                if key in out:  # a whole-weight adapter too: unusual, keep the first
                    left_out.append(key)
                    continue
                out[key] = _Sliced()
            out[key].add(sliced[0], sliced[1], module)
        else:
            out[key] = module
    if left_out:
        logging.warning("[ZenTrainer] %s: %d layer(s) don't match this model's shapes and are left out (e.g. %s)",
                        os.path.basename(path), len(left_out), left_out[0])
    if skipped is not None:
        skipped += left_out
    if not out:
        raise RuntimeError(f"{path}: no layers of this model in the training adapter")
    return out


def _fits_slice(model, key: str, sliced: tuple[int, int], out_in: tuple[int, int]) -> bool:
    """Whether a LoRA's (out, in) fits rows offset..offset+size of the fused weight it names."""
    w = model
    try:
        for part in key.split("."):
            w = getattr(w, part)
    except AttributeError:
        return False
    shape = tuple(w.shape)
    off, size = sliced
    return (len(shape) >= 2 and out_in[0] == size and off + size <= shape[0]
            and int(torch.tensor(shape[1:]).prod()) == out_in[1])


def _fits(model, key: str, out_in: tuple[int, int]) -> bool:
    """Whether a LoRA's (out, in) matches the model weight it names."""
    w = model
    try:
        for part in key.split("."):
            w = getattr(w, part)
    except AttributeError:
        return False
    shape = tuple(w.shape)
    return len(shape) >= 2 and shape[0] == out_in[0] and int(torch.tensor(shape[1:]).prod()) == out_in[1]


class Network:
    def __init__(self, patcher, spec: NetworkSpec, device: torch.device, *, bypass: bool = False,
                 init_weights: dict | None = None, assist: dict | None = None,
                 preview: dict | None = None, preview_strength: float = 1.0,
                 base: dict | None = None, base_strength: float = 1.0, wide_inputs: int = 0, scalable: bool = False):
        """`scalable`: every layer goes through _Assisted, so `strength` reaches the whole trained
        adapter (sliders); otherwise only layers with an assist, preview or base adapter are wrapped."""
        from comfy.weight_adapter import BypassInjectionManager, adapter_maps, adapters

        self.spec = spec
        self.dtype = _DTYPES[spec.lora_dtype]
        self.params: dict[str, torch.nn.Parameter] = {}
        self.modules: list[torch.nn.Module] = []
        self._bypass = BypassInjectionManager() if bypass else None
        self._injections: list = []
        init_weights = init_weights or {}
        assist = dict(assist or {})  # weight key -> frozen training adapter
        preview = dict(preview or {})  # weight key -> frozen preview (turbo) adapter
        base = dict(base or {})  # weight key -> frozen base LoRA adapter (on always)
        self.assisted = len(assist)
        self.previewed = len(preview)
        self._switch = _Switch()
        self.scalable = scalable

        alpha = float(spec.alpha if spec.alpha is not None else spec.rank)

        for name, m in patcher.model.named_modules():
            if not hasattr(m, "weight_function") or getattr(m, "weight", None) is None:
                continue
            if m.weight.ndim >= 2:
                if (not matches(name, spec.targets) or not in_blocks(name, getattr(spec, "blocks", ""))
                        or skipped(name, getattr(spec, "skip_blocks", ""))):
                    continue
                key = f"{name}.weight"
                adapter = None
                for cls in adapters:  # continuing from an existing LoRA keeps its layout, and its alpha
                    loaded = cls.load(name, init_weights, float(init_weights.get(f"{name}.alpha", alpha)),
                                      init_weights.get(f"{name}.dora_scale"))
                    if loaded is not None:
                        adapter = loaded.to_train()
                        break
                if adapter is None:
                    adapter = adapter_maps[spec.algorithm].create_train(m.weight, rank=spec.rank, alpha=alpha)
                adapter = adapter.to(device=device, dtype=self.dtype).train().requires_grad_(True)
                # alpha is the LoRA's fixed scale, a Parameter only so it saves: requires_grad_(True)
                # above made it trainable, and the optimizer drifted it (16 -> 15.99..16.03)
                if isinstance(getattr(adapter, "alpha", None), torch.nn.Parameter):
                    adapter.alpha.requires_grad_(False)
                self._own(name, adapter)
                frozen, fast, under = assist.pop(key, None), preview.pop(key, None), base.pop(key, None)
                attached = (_Assisted(adapter, frozen, self._switch, fast, preview_strength, under, base_strength)
                            if scalable or frozen is not None or fast is not None or under is not None else adapter)
                if self._bypass is not None:
                    self._bypass.add_adapter(key, attached, strength=1.0)
                else:
                    patcher.add_weight_wrapper(key, _Pinned(attached))
            elif spec.train_bias_norm:
                self._own_delta(patcher, f"{name}.weight", f"{name}.diff", m.weight.shape, device)
            if spec.train_bias_norm and getattr(m, "bias", None) is not None:
                self._own_delta(patcher, f"{name}.bias", f"{name}.diff_b", m.bias.shape, device)

        # training-adapter layers this network doesn't train still get it (e.g. the text refiner)
        for key in set(assist) | set(preview) | set(base):
            attached = _Assisted(None, assist.get(key), self._switch, preview.get(key), preview_strength,
                                 base.get(key), base_strength)
            if self._bypass is not None:
                self._bypass.add_adapter(key, attached, strength=1.0)
            else:
                patcher.add_weight_wrapper(key, _Pinned(attached))

        # channel inputs: the input layer widened for the input pictures' latents (wide_input.py)
        self.wide = None
        if wide_inputs:
            from .wide_input import WideInput, input_layer

            found = input_layer(patcher.model)
            if found is None:
                raise RuntimeError(f"{type(patcher.model).__name__} has no input layer to widen for channel inputs")
            name, layer = found
            self.wide_name, self._wide_layer = name, layer
            self.wide = WideInput(layer, wide_inputs, torch.float32, device).train()  # fp32 master, as darask0's
            if f"{name}.set_weight" in init_weights:  # continuing a channel-input LoRA
                base_w, base_b = self._wide_base()
                self.wide.load(init_weights[f"{name}.set_weight"], init_weights.get(f"{name}.bias.set_weight"), base_w, base_b)
            self.modules.append(self.wide)
            self.params[f"{name}.zt_input_delta"] = self.wide.delta
            self.params[f"{name}.zt_input_control"] = self.wide.control
            patcher.add_weight_wrapper(f"{name}.weight", self.wide.weight)
            if self.wide.bias_delta is not None:
                self.params[f"{name}.zt_input_bias"] = self.wide.bias_delta
                patcher.add_weight_wrapper(f"{name}.bias", self.wide.bias)
            logging.info("[ZenTrainer] %s widened for %d input(s): %d -> %d channels", name, wide_inputs,
                         layer.weight.shape[1], layer.weight.shape[1] * (1 + wide_inputs))

        if not self.params:
            sample = [n for n, m in patcher.model.named_modules()
                      if hasattr(m, "weight_function") and getattr(m, "weight", None) is not None
                      and m.weight.ndim >= 2][:8]
            raise RuntimeError(f"no layers matched network.targets={spec.targets!r}; "
                               f"adaptable layers look like: {', '.join(sample) or '(none found)'}")
        logging.info("[ZenTrainer] %d adapters, %d tensors (%s, rank %d, alpha %g)",
                     len(self.modules), len(self.params), spec.algorithm, spec.rank, alpha)

    def _own(self, name: str, adapter: torch.nn.Module) -> None:
        self.modules.append(adapter)
        for pname, p in adapter.named_parameters():
            self.params[f"{name}.{pname}"] = p

    def _own_delta(self, patcher, key: str, param_name: str, shape, device) -> None:
        mod = _BiasDiff(shape, self.dtype, self._switch if self.scalable else None).to(device).train()
        self.modules.append(mod)
        self.params[param_name] = mod.diff
        patcher.add_weight_wrapper(key, _Pinned(mod))

    def inject(self, patcher) -> None:
        """Bypass mode only: install the forward hooks. Pair with eject()."""
        if self._bypass is not None and not self._injections:
            self._injections = self._bypass.create_injections(patcher.model)
            for inj in self._injections:
                inj.inject(patcher)

    def eject(self, patcher) -> None:
        for inj in self._injections:
            inj.eject(patcher)
        self._injections = []

    @property
    def assist(self) -> bool:
        """Whether the training adapter (if any) is applied: off for samples."""
        return self._switch.on

    @assist.setter
    def assist(self, on: bool) -> None:
        self._switch.on = on

    @property
    def strength(self) -> float:
        """How much of the trained adapter applies (scalable networks): 1 as trained, 0 the base
        model, -1 the opposite direction."""
        return self._switch.strength

    @strength.setter
    def strength(self, s: float) -> None:
        self._switch.strength = float(s)

    @contextlib.contextmanager
    def at_strength(self, s: float):
        before = self._switch.strength
        self._switch.strength = float(s)
        try:
            yield
        finally:
            self._switch.strength = before

    @contextlib.contextmanager
    def disabled(self):
        """The base model for a moment (a "without the LoRA" sample): every trained tensor is
        zeroed, which makes each adapter add nothing (LoRA, LoHa, LoKr, OFT, diffs), then
        restored exactly."""
        saved = {k: p.detach().clone() for k, p in self.params.items()}
        try:
            with torch.no_grad():
                for p in self.params.values():
                    p.zero_()
            yield
        finally:
            with torch.no_grad():
                for k, p in self.params.items():
                    p.copy_(saved[k])

    def trainable(self) -> list[torch.nn.Parameter]:
        return [p for p in self.params.values() if p.requires_grad]

    def param_groups(self, lr: float, lora_plus: float = 0.0) -> list:
        """The trainable tensors, or groups: with LoRA+ the up (B) matrices at lr × lora_plus, with
        channel inputs the input layer's new columns at lr × CONTROL_LR_SCALE."""
        control = [self.wide.control] if self.wide is not None else []
        up = [p for k, p in self.params.items() if p.requires_grad and ".lora_up." in f"{k}."] if lora_plus > 1 else []
        rest = [p for p in self.trainable() if not any(p is q for q in up + control)]
        if not up and not control:
            return self.trainable()
        groups = [{"params": rest, "lr": lr}]
        if up:  # LoHa, LoKr, OFT have no up matrices to speed up
            groups.append({"params": up, "lr": lr * lora_plus, "lr_mult": lora_plus})
        if control:
            from .wide_input import CONTROL_LR_SCALE

            groups.append({"params": control, "lr": lr * CONTROL_LR_SCALE, "lr_mult": CONTROL_LR_SCALE})
        return groups

    def _wide_base(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The widened layer's own (base) weight and bias, full precision on the CPU."""
        def plain(t):
            if t is None:
                return None
            t = t.dequantize() if hasattr(t, "dequantize") else t
            return t.detach().to("cpu", torch.float32)
        return plain(self._wide_layer.weight), plain(getattr(self._wide_layer, "bias", None))

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Tensors in ComfyUI's LoRA-file layout, detached on CPU. A widened input layer is saved
        whole, as ComfyUI's "set" patches."""
        out = {k: v.detach().to("cpu", self.dtype).contiguous() for k, v in self.params.items() if ".zt_input_" not in k}
        if self.wide is not None:
            w, b = self.wide.full(*self._wide_base())
            out[f"{self.wide_name}.set_weight"] = w.to(torch.bfloat16).contiguous()
            if b is not None:
                out[f"{self.wide_name}.bias.set_weight"] = b.to(torch.bfloat16).contiguous()
        return out
