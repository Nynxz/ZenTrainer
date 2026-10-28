// Generated from engine/zentrainer/spec.py: `python -m zentrainer.spec --ts > web/src/api/engine/spec.fields.ts`
// in engine/. Don't edit: spec.ts checks its TrainSpec against these at type-check time.

export interface SpecFields {
  spec_version: number
  name: string
  seed: number
  model: {
    kind: "checkpoint" | "split"
    ckpt_name: string | null
    unet_name: string | null
    unet_weight_dtype: string
    clip_name: string | null
    clip_type: string | null
    vae_name: string | null
    audio_vae_name: string | null
  }
  dataset: {
    folder: string
    resolution: number
    stages: Record<string, unknown>[]
    stage_blend: number
    caption_ext: string
    buckets: boolean
    bucket_step: number
    edit: boolean
    control_sets: string[]
    edit_method: string
    control_dropout: number
    edit_template: string
    edit_grounded: boolean
    disk_cache: boolean
    video_keyframes: string
    caption_dropout: number
    caption_set: string
    caption_mix: string[]
    trigger: string
    trigger_prepend: boolean
    masked_loss: boolean
    unmasked_weight: number
    video_frames: number
    video_fps: number
    video_mode: string
    video_clips: number
    video_resolution: number
    video_audio: boolean
    video_avoid_cuts: boolean
    video_stills: boolean
  }
  network: {
    algorithm: string
    rank: number
    alpha: number | null
    lora_dtype: string
    targets: string | string[]
    train_bias_norm: boolean
    init_from: string | null
    training_adapter: string | null
    base_lora: string | null
    base_strength: number
    blocks: string
    skip_blocks: string
  }
  optim: {
    optimizer: string
    learning_rate: number
    weight_decay: number
    lr_scheduler: string
    warmup_steps: number
    max_grad_norm: number
    steps: number
    batch_size: number
    grad_accumulation_steps: number
    lora_plus: number
    ema: number
    lr_floor: number
  }
  objective: {
    loss: string
    target: string
    timesteps: string
    timestep_scheduler: string
    timestep_scheduler_steps: number
    logit_mean: number
    logit_std: number
    uniform_mix: number
    shift: number
    bias: string
    bias_power: number
    weighting: string
    spread: boolean
  }
  memory: {
    training_dtype: string
    gradient_checkpointing: boolean
    grad_checkpoint_depth: number
    grad_checkpoint_skip: number | "auto"
    offloading: boolean
    vram_headroom: number
    tread: number
    tread_steps: number
    int8_forward: boolean
    int8_backward: string
    fp8_backward: boolean
    compile_blocks: boolean | "auto"
    fused_qk_rope: boolean | "auto"
    bypass_mode: boolean
    attention: string
  }
  samples: {
    likeness: boolean
    every: number
    prompts: (string | Record<string, unknown>)[]
    negative: string
    width: number
    height: number
    frames: number
    fps: number
    steps: number
    cfg: number
    sampler: string
    scheduler: string
    sigmas: number[]
    seed: number
    baseline: boolean
    model_name: string | null
    model_weight_dtype: string
    lora_strength: number
    turbo_lora: string | null
    turbo_strength: number
    slg_mode: string
    slg_layers: number[]
    slg_double_layers: number[]
    slg_scale: number
    slg_start: number
    slg_end: number
  }
  validation: {
    every: number
    images: number
    noise_levels: number
    seed: number
    dataset: string
  }
  output: {
    save_every: number
    lora_dir: string | null
    keep_states: string
    save_dtype: string
    keep: string
    keep_count: number
  }
  slider: {
    kind: string
    pair_set: string
    dataset_end: string
    positive: string
    negative: string
    neutral: string
    anchors: string[]
    guidance: number
    anchor_weight: number
    directions: string
  }
}
