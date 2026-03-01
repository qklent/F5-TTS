# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F5-TTS is a flow-matching TTS model. This repository is on the `mask_fill` branch, which extends the base model with phoneme-level mask filling for speech correction (e.g., correcting specific phoneme disorders). The key custom additions are `CFM.forward()` supporting a `phoneme_mask` parameter, a custom dataset class, and a dedicated training script.

## Installation

```bash
pip install -e .
```

Requires PyTorch and torchaudio installed separately with the appropriate device backend (CUDA/ROCm/MPS).

## Common Commands

### Linting
```bash
pre-commit run --all-files
```
Config: `ruff.toml` (line-length=120, target python 3.10+). Some model files have `# ruff: noqa: F722 F821` to allow tensor annotation syntax.

### Inference
```bash
# CLI inference
f5-tts_infer-cli --model F5TTS_v1_Base --ref_audio ref.wav --ref_text "..." --gen_text "..."

# Gradio web UI
f5-tts_infer-gradio
```

### Standard Training (with Hydra configs)
```bash
accelerate config  # one-time setup
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml
```

### Mask Fill Training (this branch)
```bash
# Via Hydra config (recommended)
python -m f5_tts.train.train_mask_fill --config-name F5TTS_MaskFill

# Override config values
python -m f5_tts.train.train_mask_fill --config-name F5TTS_MaskFill \
    datasets.path=data/my_dataset \
    datasets.mask_key=hard_s_timestamps \
    optim.epochs=50
```

### Finetuning via Gradio
```bash
f5-tts_finetune-gradio
```

## Architecture

### Core Flow
The model is a **Conditional Flow Matching (CFM)** model. `CFM` (`src/f5_tts/model/cfm.py`) wraps a transformer backbone and provides:
- `forward()` — training: computes flow matching loss over masked mel-spectrogram regions
- `sample()` — inference: runs ODE integration to generate audio from noise

### Transformer Backbones (`src/f5_tts/model/backbones/`)
- `DiT` — Diffusion Transformer with ConvNeXt V2 (used by F5-TTS)
- `UNetT` — Flat-UNet Transformer (used by E2-TTS)
- `MMDiT` — Multi-Modal DiT variant

### Mask Fill Extension (this branch)
The `phoneme_mask` parameter was added to `CFM.forward()`:
- When `phoneme_mask` is provided (a `[batch, seq_len]` bool tensor, `True` = mask this frame), it overrides the random span masking used in standard training.
- When `None`, falls back to original random masking behavior (fully backward compatible).

Key files added/modified for this feature:
- `src/f5_tts/model/dataset_masked_phoneme_optimized_fixed.py` — `MemoryOptimizedMaskedPhonemeDataset` and `collate_fn_masked`: loads HuggingFace datasets with phoneme timestamp annotations, converts timestamps to mel-frame masks with a configurable margin
- `src/f5_tts/train/train_mask_fill.py` — `MaskedPhonemeTrainer` extends `Trainer`, passing `phoneme_mask` from batch to `CFM.forward()`
- `src/f5_tts/configs/F5TTS_MaskFill.yaml` — Hydra config for mask fill fine-tuning

### Training Infrastructure
- `Trainer` (`src/f5_tts/model/trainer.py`) uses HuggingFace `accelerate` for distributed training and EMA
- `DynamicBatchSampler` in `src/f5_tts/model/dataset.py` supports frame-wise batching
- Configs use Hydra (`src/f5_tts/configs/*.yaml`); `train.py` and `train_mask_fill.py` use `@hydra.main`
- Checkpoints saved to `ckpts/{save_dir}/`; pretrained checkpoints prefixed with `pretrained_`

### Dataset Format for Mask Fill
HuggingFace dataset with fields:
- `audio`: `{'array': [...], 'sampling_rate': int}`
- `text`: transcription string
- `<mask_key>` (e.g., `hard_s_timestamps`): list of `[start_sec, end_sec]` pairs

### Inference with Masks
For correcting specific phoneme regions at inference, use `CFM.sample()` with `edit_mask` (a `[batch, seq_len]` bool tensor where `True` = keep original, `False` = regenerate).

### Vocoders
`vocos` (default) and `bigvgan` are supported. Mel spec parameters: 24kHz sample rate, 100 mel channels, hop_length=256, n_fft=1024.
