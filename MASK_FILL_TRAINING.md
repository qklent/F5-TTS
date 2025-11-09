# Phoneme-Based Mask Filling Training for F5-TTS

This guide explains how to fine-tune F5-TTS for speech correction tasks using phoneme-based mask filling. This approach is designed for correcting specific speech disorders by learning to fill in masked phoneme segments.

## Use Case

This training method is ideal for:
- Correcting specific phoneme pronunciation issues (e.g., hard 'S' sounds)
- Speech therapy applications
- Personalized speech correction models
- Any task requiring targeted phoneme-level editing

## How It Works

1. **Training Phase**:
   - The model is trained on clean speech data with correct phoneme pronunciation
   - Specific phonemes (e.g., hard 'S' sounds) are masked during training
   - The model learns to predict/fill these masked regions from context

2. **Inference Phase**:
   - Your speech with incorrect phonemes is provided
   - The incorrect phoneme regions are masked
   - The model fills these regions with correctly pronounced phonemes

## Dataset Requirements

Your dataset must include:
- `audio`: Audio data (HuggingFace AudioDecoder format)
- `text`: Corresponding text transcription
- `hard_s_timestamps` (or custom key): List of `[start, end]` time pairs in seconds for phonemes to mask

Example dataset structure (as shown in `data_preparation.ipynb`):
```python
{
    'audio': {'array': [...], 'sampling_rate': 44100},
    'text': 'Вот только они совсем не радовали...',
    'hard_s_timestamps': [[1.05, 1.15], [3.62, 3.79], [4.60, 4.75]],
    # Optional: other metadata
}
```

## Data Preparation

See `data_preparation.ipynb` for an example of how to:
1. Load a dataset with phoneme timestamps
2. Extract specific phoneme timestamps (e.g., hard 'S' sounds)
3. Verify the timestamps align with audio segments

## Training

### Basic Usage

```bash
python -m f5_tts.train.train_mask_fill \
    --dataset_path data/test_processed \
    --exp_name F5TTS_v1_Base \
    --finetune \
    --batch_size_per_gpu 3200 \
    --epochs 10
```

### Full Options

```bash
python -m f5_tts.train.train_mask_fill \
    --dataset_path data/your_dataset \
    --mask_key hard_s_timestamps \
    --exp_name F5TTS_v1_Base \
    --dataset_name my_speech_correction \
    --finetune \
    --pretrain path/to/checkpoint.pt \
    --batch_size_per_gpu 3200 \
    --batch_size_type frame \
    --max_samples 64 \
    --epochs 50 \
    --learning_rate 1e-5 \
    --num_warmup_updates 2000 \
    --save_per_updates 5000 \
    --keep_last_n_checkpoints 3 \
    --tokenizer char \
    --num_workers 4 \
    --logger wandb \
    --log_samples
```

### Key Parameters

- `--dataset_path`: Path to your HuggingFace dataset (required)
- `--mask_key`: Key in dataset containing phoneme timestamps (default: `hard_s_timestamps`)
- `--exp_name`: Model architecture to use (`F5TTS_v1_Base`, `F5TTS_Base`, or `E2TTS_Base`)
- `--finetune`: Start from pretrained checkpoint
- `--pretrain`: Path to custom pretrained checkpoint (optional)
- `--batch_size_per_gpu`: Batch size in frames (default: 3200)
- `--batch_size_type`: `frame` or `sample` (default: `frame`)
- `--epochs`: Number of training epochs
- `--tokenizer`: `char`, `pinyin`, or `custom` (default: `char` for Russian/multilingual)

## Implementation Details

### Files Created

1. **`src/f5_tts/model/dataset_masked_phoneme.py`**
   - `MaskedPhonemeDataset`: Custom dataset class that loads phoneme timestamps
   - `collate_fn_masked`: Collate function that handles phoneme masks in batches

2. **`src/f5_tts/model/cfm.py`** (modified)
   - Added `phoneme_mask` parameter to `forward()` method
   - When provided, uses phoneme-specific masking instead of random masking

3. **`src/f5_tts/train/train_mask_fill.py`**
   - Training script with `MaskedPhonemeTrainer` class
   - Passes phoneme masks from dataset to model during training

### What Changed in CFM

The CFM model's forward pass now supports an optional `phoneme_mask` parameter:
- When `phoneme_mask` is provided: Uses it to mask specific phonemes
- When `phoneme_mask` is `None`: Falls back to random masking (original behavior)

This is backward compatible with existing training scripts.

## Tips for Best Results

1. **Dataset Quality**: Use clean recordings with correct pronunciation
2. **Timestamp Accuracy**: Ensure phoneme timestamps are accurate (±50ms tolerance)
3. **Data Quantity**: More examples of the target phoneme = better results
   - Recommended: At least 1000+ examples of each phoneme to correct
4. **Fine-tuning**: Always start from a pretrained checkpoint (`--finetune`)
5. **Learning Rate**: Use small learning rates (1e-5 to 1e-6) for fine-tuning
6. **Batch Size**: Adjust based on GPU memory (3200 frames works well on 16GB+ GPU)

## Monitoring Training

### With WandB
```bash
python -m f5_tts.train.train_mask_fill \
    --dataset_path data/test_processed \
    --finetune \
    --logger wandb \
    --log_samples
```

### With TensorBoard
```bash
python -m f5_tts.train.train_mask_fill \
    --dataset_path data/test_processed \
    --finetune \
    --logger tensorboard

# In another terminal
tensorboard --logdir runs/
```

## Checkpoints

Checkpoints are saved to `ckpts/{dataset_name}/`:
- `model_last.pt`: Latest checkpoint (saved every `--last_per_updates`)
- `model_{update}.pt`: Periodic checkpoints (saved every `--save_per_updates`)
- `pretrained_*.pt`: Original pretrained model (not deleted during rotation)

## Troubleshooting

### Out of Memory
- Reduce `--batch_size_per_gpu` (try 1600, 800, etc.)
- Reduce `--max_samples` (try 32, 16)
- Use `--grad_accumulation_steps 2` or higher

### Dataset Format Issues
- Ensure your dataset is a HuggingFace Dataset (use `datasets.load_from_disk()`)
- Verify the `mask_key` exists in your dataset
- Check that timestamps are in seconds (not frames)

### No Phonemes Being Masked
- Verify timestamps are in the correct format: `[[start1, end1], [start2, end2], ...]`
- Check that timestamps align with mel-spectrogram frames
- Ensure timestamps are not empty (`[]`)

## Next Steps

After training:
1. Find your best checkpoint in `ckpts/{dataset_name}/`
2. Use the model for inference with the `edit_mask` parameter in `CFM.sample()`
3. Provide your speech with a mask indicating where to correct phonemes

For inference examples, refer to the F5-TTS inference documentation.
