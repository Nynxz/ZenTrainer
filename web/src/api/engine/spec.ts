// Run specs: the TrainSpec the New run page builds, its defaults, presets. Mirrors engine/zentrainer/spec.py; keep them in step.

export interface TrainSpec {
  /** the spec's layout (engine spec.SPEC_VERSION): the engine brings older ones up to date */
  spec_version: number
  name: string
  seed: number
  model: {
    kind: 'checkpoint' | 'split'
    ckpt_name?: string | null
    unet_name?: string | null
    unet_weight_dtype?: string
    clip_name?: string | null
    clip_type?: string | null
    vae_name?: string | null
    /** joint audio-video models: the audio VAE (sound in training and samples) */
    audio_vae_name?: string | null
  }
  dataset: {
    folder: string
    resolution: number
    /** lower resolutions first: [{resolution: 768, until: 0.6}] trains at 768 for 60% of the run */
    stages?: { resolution: number; until: number }[]
    /** share of the run each resolution change blends over */
    stage_blend?: number
    caption_ext: string
    buckets: boolean
    bucket_step: number
    caption_dropout: number
    masked_loss: boolean
    edit?: boolean
    /** control sets (named layers of input pictures) the run edits from, in order; any turns edit on */
    control_sets?: string[]
    /** "concat": channel inputs (aligned pictures as extra latent channels, MOSS Image) */
    edit_method?: string
    /** channel inputs: share of steps each input is left out */
    control_dropout?: number
    edit_template?: string
    edit_grounded?: boolean
    /** each picture's latents in a file of its own, read as training uses them: datasets bigger than RAM */
    disk_cache?: boolean
    /** MiniMax H3 image-to-video: each clip conditioned on its first (and last) frame */
    video_keyframes?: string
    /** each clip also trains its sharpest frame as a photo */
    video_stills?: boolean
    unmasked_weight: number
    /** videos: longest clip (snapped to the VAE), fps (0 = source), which windows, how many, size */
    video_frames: number
    video_fps: number
    video_mode: 'uniform' | 'head' | 'chunks'
    video_clips: number
    video_resolution: number
    /** keep clips inside one shot (hard cuts detected) */
    video_avoid_cuts?: boolean
    /** train on each clip's soundtrack (joint audio-video models) */
    video_audio?: boolean
    caption_set: string
    caption_mix: string[]
    trigger: string
    trigger_prepend: boolean
  }
  network: {
    algorithm: string
    rank: number
    alpha: number | null
    lora_dtype: string
    targets: 'attn' | 'mlp' | 'attn_mlp' | 'all' | string[]
    train_bias_norm: boolean
    init_from: string | null
    /** a LoRA applied while training only (not in samples or the saved file): a de-distillation adapter */
    training_adapter: string | null
    /** a LoRA the trained one sits on everywhere (training, samples, use); saved beside, not in */
    base_lora?: string | null
    base_strength?: number
    /** only these blocks get adapters, e.g. "20-49" ("" = all): the backward stops at the first */
    blocks?: string
    /** these blocks get no adapters, e.g. "22" or "22, 30-31" */
    skip_blocks?: string
  }
  optim: {
    optimizer: string
    learning_rate: number
    weight_decay: number
    /** the learning rate over the run */
    lr_scheduler: 'constant' | 'linear' | 'cosine'
    warmup_steps: number
    max_grad_norm: number
    steps: number
    batch_size: number
    grad_accumulation_steps: number
    /** LoRA+: up matrices learn this many times faster (0 = off) */
    lora_plus?: number
    ema?: number
    /** Automagic v3's and v4's lowest LR, as a fraction of the start LR; 0 = none */
    lr_floor?: number
  }
  objective: {
    loss: 'mse' | 'l1' | 'huber' | 'smooth_l1'
    target: 'native' | 'x0'
    timesteps: 'uniform' | 'logit_normal' | 'scheduler'
    /** timesteps "scheduler": which of ComfyUI's, and at how many steps */
    timestep_scheduler?: string
    timestep_scheduler_steps?: number
    logit_mean: number
    logit_std: number
    /** logit-normal: share of steps drawn uniformly instead */
    uniform_mix?: number
    /** flow models: training timestep shift (0 = the model's own sampling shift) */
    shift: number
    /** lean the noise levels to one end (AI Toolkit's content / style) */
    bias?: 'balanced' | 'high_noise' | 'low_noise'
    /** how hard: the draw's power (3 = AI Toolkit's cubic) */
    bias_power?: number
    /** per-noise-level loss weights: "flex" = AI Toolkit's "weighted" (flow models), "balance"
     * = measured on the model as it trains, evening out every noise level's pull */
    weighting?: '' | 'flex' | 'balance'
    /** noise levels from an even sequence instead of random draws */
    spread?: boolean
  }
  memory: {
    training_dtype: 'bf16' | 'fp16' | 'fp32'
    gradient_checkpointing: boolean
    checkpoint_depth: number
    /** the last N blocks keep their activations: faster, more VRAM */
    checkpoint_skip: number
    /** TREAD: share of a video clip's tokens routed around the middle blocks (0 = off) */
    tread?: number
    /** int8 ConvRot bases' backward: "w8a8" on int8 tensor cores (engine default), "fused" Triton at
     *  full precision, "bf16" the weight rebuilt (slowest) */
    int8_backward?: 'w8a8' | 'fused' | 'bf16'
    /** torch.compile the main blocks; unset/null = auto (long runs, with Triton) */
    compile_blocks?: boolean | null
    /** Krea 2 / MOSS: the q/k norms and rope as one op that trains (experimental, engine default off) */
    /** the fused q/k norm + rope op; unset/null = auto (on while the blocks run uncompiled) */
    fused_qk_rope?: boolean | null
    /** int8 bases: frozen layers run the int8 kernel forward (engine default on) */
    int8_forward?: boolean
    tread_steps?: number
    offloading: boolean
    /** fp8 bases: the backward's matmuls in fp8 too */
    fp8_backward: boolean
    bypass_mode: boolean
    attention: string
  }
  samples: {
    every: number
    /** a prompt, or a sample with its own size/length/seed */
    prompts: (string | SamplePrompt)[]
    negative: string
    width: number
    height: number
    /** video models: frames per sample (1 = a still) */
    frames?: number
    fps?: number
    steps: number
    cfg: number
    sampler: string
    scheduler: string
    /** custom sigmas, high to low, in place of the scheduler and steps; [] = the scheduler */
    sigmas?: number[]
    seed: number
    baseline: boolean
    model_name: string | null
    model_weight_dtype: string
    lora_strength: number
    /** a turbo LoRA under every sample (few-step previews), and its strength */
    turbo_lora?: string | null
    turbo_strength?: number
    /** skip-layer guidance: off, skip the layers in the negative pass, or an extra pass without them */
    slg_mode: 'off' | 'uncond' | 'standard'
    slg_layers: number[]
    slg_double_layers: number[]
    slg_scale: number
    slg_start: number
    slg_end: number
  }
  validation: { every: number; images: number; timesteps: number; seed: number }
  /** where checkpoints go, which resume states are kept, and which scheduled checkpoints
   *  (keep: all, the keep_count latest, or the keep_count best by likeness or validation loss;
   *  engine harness/keep.py) */
  output: {
    /** a checkpoint every this many steps; 0 = only the final LoRA */
    save_every: number
    lora_dir: string | null
    keep_states: 'latest' | 'all' | 'none'
    save_dtype: 'bf16' | 'fp16' | 'fp32'
    keep: 'all' | 'latest' | 'likeness' | 'val_loss' | 'likeness_new_best' | 'val_loss_new_best'
    keep_count: number
  }
  /** a slider LoRA (engine harness/slider.py): its strength moves pictures from `negative` (-1)
   *  to `positive` (+1), 0 the base model. "text": made from the prompts; the dataset's pictures
   *  are only where the noise starts, so their captions aren't used */
  slider: {
    kind: '' | 'pairs' | 'text'
    /** pairs: the control set holding each picture's partner, and which end the dataset's own are */
    pair_set: string
    dataset_end: 'positive' | 'negative'
    positive: string
    negative: string
    neutral: string
    /** prompts the slider should leave alone */
    anchors: string[]
    guidance: number
    anchor_weight: number
    directions: 'both' | 'positive'
  }
}

export interface Preset<T = unknown> {
  id: string
  name: string
  description: string
  data: T
  builtin: boolean
}

export type PresetKind = 'run' | 'prompts'

/** a sample with its own settings (unset ones follow the run's) */
export interface SamplePrompt {
  prompt: string
  width?: number
  height?: number
  frames?: number
  seed?: number
  /** edit runs: the picture to edit (dataset-relative) */
  control?: string
}

export const promptText = (p: string | SamplePrompt) => (typeof p === 'string' ? p : p.prompt)

export function defaultSpec(): TrainSpec {
  return {
    spec_version: 2,
    name: 'my-lora',
    seed: 0,
    model: { kind: 'checkpoint', ckpt_name: null, unet_weight_dtype: 'default' },
    dataset: {
      folder: '',
      resolution: 1024,
      caption_ext: '.txt',
      buckets: true,
      bucket_step: 64,
      caption_dropout: 0,
      masked_loss: false,
      unmasked_weight: 0,
      video_frames: 49,
      video_fps: 0,
      video_mode: 'uniform',
      video_clips: 1,
      video_resolution: 512,
      caption_set: 'default',
      caption_mix: [],
      trigger: '',
      trigger_prepend: true,
    },
    network: {
      algorithm: 'LoRA',
      rank: 16,
      alpha: null,
      lora_dtype: 'fp32',
      targets: 'attn_mlp',
      train_bias_norm: false,
      init_from: null,
      training_adapter: null,
    },
    optim: {
      optimizer: 'AdamW',
      learning_rate: 1e-4,
      weight_decay: 0,
      lr_scheduler: 'constant',
      warmup_steps: 0,
      max_grad_norm: 1,
      steps: 1000,
      batch_size: 1,
      grad_accumulation_steps: 1,
    },
    objective: { loss: 'mse', target: 'native', timesteps: 'uniform', logit_mean: 0, logit_std: 1, shift: 0 },
    memory: {
      training_dtype: 'bf16',
      gradient_checkpointing: true,
      checkpoint_depth: 1,
      checkpoint_skip: 0,
      offloading: false,
      fp8_backward: false,
      bypass_mode: false,
      attention: 'pytorch',
    },
    samples: {
      every: 250, // with each save (output.save_every's default): each checkpoint gets samples, and a likeness
      prompts: [],
      negative: '',
      width: 1024,
      height: 1024,
      steps: 20,
      cfg: 5,
      sampler: 'euler',
      scheduler: 'normal',
      sigmas: [],
      seed: 42,
      baseline: true,
      model_name: null,
      model_weight_dtype: 'default',
      lora_strength: 1,
      slg_mode: 'off',
      slg_layers: [],
      slg_double_layers: [],
      slg_scale: 3,
      slg_start: 0.01,
      slg_end: 1,
    },
    validation: { every: 100, images: 4, timesteps: 4, seed: 1234 },
    output: { save_every: 250, lora_dir: null, keep_states: 'latest', save_dtype: 'bf16', keep: 'all', keep_count: 3 },
    slider: { kind: '', pair_set: '', dataset_end: 'positive', positive: '', negative: '', neutral: '', anchors: [], guidance: 3, anchor_weight: 1, directions: 'both' },
  }
}
