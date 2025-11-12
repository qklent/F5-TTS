#!/bin/bash

# Config-based mask fill training with F5-TTS
# Example usage for mask fill fine-tuning

python -m f5_tts.train.train_mask_fill --config-name=F5TTS_MaskFill \
    datasets.path=qklent/tonebooks-mfa-phonemes-only-hard-s \
    datasets.mask_key=hard_s_timestamps \
    ckpts.logger=wandb \
    ckpts.log_samples=true \
    optim.epochs=10

# You can also override other config values:
# python -m f5_tts.train.train_mask_fill --config-name=F5TTS_MaskFill \
#     datasets.path=../data/test_processed \
#     datasets.mask_key=hard_s_timestamps \
#     model.tokenizer=char \
#     model.name=F5TTS_Base_MaskFill \
#     datasets.batch_size_per_gpu=1600 \
#     optim.learning_rate=5e-6 \
#     ckpts.finetune=true \
#     ckpts.pretrain_path=/path/to/custom/checkpoint.pt