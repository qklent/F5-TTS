"""
Training script for phoneme-based mask filling fine-tuning.
Specifically designed for correcting speech disorders by learning to fill masked phonemes.

Example usage:
python -m f5_tts.train.train_mask_fill \\
    --dataset_path ../data/test_processed \\
    --exp_name F5TTS_v1_Base \\
    --finetune \\
    --batch_size_per_gpu 3200 \\
    --epochs 10
"""

import argparse
import os
import shutil
from importlib.resources import files

from cached_path import cached_path
from datasets import load_from_disk

from f5_tts.model import CFM, DiT, UNetT
from f5_tts.model.dataset_masked_phoneme import MaskedPhonemeDataset, collate_fn_masked
from f5_tts.model.utils import get_tokenizer


# -------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'


# -------------------------- Argument Parsing --------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train F5-TTS with Phoneme-Based Mask Filling")

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to HuggingFace dataset with phoneme timestamps (e.g., ../data/test_processed)",
    )
    parser.add_argument(
        "--mask_key",
        type=str,
        default="hard_s_timestamps",
        help="Key in dataset containing phoneme timestamps to mask (default: hard_s_timestamps)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="F5TTS_v1_Base",
        choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"],
        help="Experiment name",
    )
    parser.add_argument("--dataset_name", type=str, default="phoneme_masked", help="Name for saving checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--batch_size_per_gpu", type=int, default=3200, help="Batch size per GPU (frames)")
    parser.add_argument(
        "--batch_size_type", type=str, default="frame", choices=["frame", "sample"], help="Batch size type"
    )
    parser.add_argument("--max_samples", type=int, default=64, help="Max sequences per batch")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_warmup_updates", type=int, default=2000, help="Warmup updates")
    parser.add_argument("--save_per_updates", type=int, default=5000, help="Save checkpoint every N updates")
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=3,
        help="-1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints",
    )
    parser.add_argument("--last_per_updates", type=int, default=1000, help="Save last checkpoint every N updates")
    parser.add_argument("--finetune", action="store_true", help="Use pretrained checkpoint for fine-tuning")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to custom pretrained checkpoint")
    parser.add_argument(
        "--tokenizer", type=str, default="char", choices=["pinyin", "char", "custom"], help="Tokenizer type"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to custom tokenizer vocab file (only used if tokenizer = 'custom')",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Log inferenced samples per ckpt save updates",
    )
    parser.add_argument("--logger", type=str, default=None, choices=[None, "wandb", "tensorboard"], help="logger")
    parser.add_argument(
        "--bnb_optimizer",
        action="store_true",
        help="Use 8-bit Adam optimizer from bitsandbytes",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")

    return parser.parse_args()


# -------------------------- Main Training -------------------------- #
def main():
    args = parse_args()

    checkpoint_path = str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}"))

    # Model parameters based on experiment name
    if args.exp_name == "F5TTS_v1_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
            else:
                ckpt_path = args.pretrain

    elif args.exp_name == "F5TTS_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    elif args.exp_name == "E2TTS_Base":
        wandb_resume_id = None
        model_cls = UNetT
        model_cfg = dict(
            dim=1024,
            depth=24,
            heads=16,
            ff_mult=4,
            text_mask_padding=False,
            pe_attn_head=1,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    # Copy pretrained checkpoint if fine-tuning
    if args.finetune:
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

        file_checkpoint = os.path.basename(ckpt_path)
        if not file_checkpoint.startswith("pretrained_"):
            file_checkpoint = "pretrained_" + file_checkpoint
        file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
        if not os.path.isfile(file_checkpoint):
            shutil.copy2(ckpt_path, file_checkpoint)
            print(f"Copied pretrained checkpoint to {file_checkpoint}")

    # Setup tokenizer
    tokenizer = args.tokenizer
    if tokenizer == "custom":
        if not args.tokenizer_path:
            raise ValueError("Custom tokenizer selected, but no tokenizer_path provided.")
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = args.dataset_name

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    print(f"\nVocab size: {vocab_size}")
    print(f"Vocoder: {mel_spec_type}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Mask key: {args.mask_key}")

    # Mel spectrogram kwargs
    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    # Create model
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    # Load datasets
    print("\nLoading datasets...")
    hf_dataset = load_from_disk(args.dataset_path)

    # Check if dataset has train/validation splits
    if hasattr(hf_dataset, "keys"):
        train_hf = hf_dataset["train"]
        print(f"Train samples: {len(train_hf)}")
        if "validation" in hf_dataset.keys():
            print(f"Validation samples: {len(hf_dataset['validation'])}")
    else:
        train_hf = hf_dataset
        print(f"Total samples: {len(train_hf)}")

    # Create custom dataset
    train_dataset = MaskedPhonemeDataset(
        train_hf,
        mask_key=args.mask_key,
        **mel_spec_kwargs,
    )

    print(f"\nDataset loaded successfully with {len(train_dataset)} samples")

    # Import and create trainer with custom collate function
    from f5_tts.model import Trainer

    # We need to monkey-patch the trainer to use our custom collate function
    # Create a custom Trainer class
    class MaskedPhonemeTrainer(Trainer):
        """Custom trainer that passes phoneme_mask to the model."""

        def train(self, train_dataset, num_workers=16, resumable_with_seed=None):
            # Import here to avoid circular dependency
            import math

            import torch
            import torchaudio
            from torch.utils.data import DataLoader, SequentialSampler
            from tqdm import tqdm

            from f5_tts.model.dataset import DynamicBatchSampler
            from f5_tts.model.utils import exists

            if self.log_samples:
                from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

                vocoder = load_vocoder(
                    vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
                )
                target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
                log_samples_path = f"{self.checkpoint_path}/samples"
                os.makedirs(log_samples_path, exist_ok=True)

            if exists(resumable_with_seed):
                generator = torch.Generator()
                generator.manual_seed(resumable_with_seed)
            else:
                generator = None

            # Use custom collate function
            if self.batch_size_type == "sample":
                train_dataloader = DataLoader(
                    train_dataset,
                    collate_fn=collate_fn_masked,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True,
                    batch_size=self.batch_size_per_gpu,
                    shuffle=True,
                    generator=generator,
                )
            elif self.batch_size_type == "frame":
                self.accelerator.even_batches = False
                sampler = SequentialSampler(train_dataset)
                batch_sampler = DynamicBatchSampler(
                    sampler,
                    self.batch_size_per_gpu,
                    max_samples=self.max_samples,
                    random_seed=resumable_with_seed,
                    drop_residual=False,
                )
                train_dataloader = DataLoader(
                    train_dataset,
                    collate_fn=collate_fn_masked,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True,
                    batch_sampler=batch_sampler,
                )
            else:
                raise ValueError(
                    f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}"
                )

            from torch.optim.lr_scheduler import LinearLR, SequentialLR

            warmup_updates = self.num_warmup_updates * self.accelerator.num_processes
            total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
            decay_updates = total_updates - warmup_updates
            warmup_scheduler = LinearLR(
                self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates
            )
            decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
            self.scheduler = SequentialLR(
                self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
            )
            train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)
            start_update = self.load_checkpoint()
            global_update = start_update

            if exists(resumable_with_seed):
                orig_epoch_step = len(train_dataloader)
                start_step = start_update * self.grad_accumulation_steps
                skipped_epoch = int(start_step // orig_epoch_step)
                skipped_batch = start_step % orig_epoch_step
                skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
            else:
                skipped_epoch = 0

            for epoch in range(skipped_epoch, self.epochs):
                self.model.train()
                if exists(resumable_with_seed) and epoch == skipped_epoch:
                    progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                    current_dataloader = skipped_dataloader
                else:
                    progress_bar_initial = 0
                    current_dataloader = train_dataloader

                if hasattr(train_dataloader, "batch_sampler") and hasattr(
                    train_dataloader.batch_sampler, "set_epoch"
                ):
                    train_dataloader.batch_sampler.set_epoch(epoch)

                progress_bar = tqdm(
                    range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                    desc=f"Epoch {epoch + 1}/{self.epochs}",
                    unit="update",
                    disable=not self.accelerator.is_local_main_process,
                    initial=progress_bar_initial,
                )

                for batch in current_dataloader:
                    with self.accelerator.accumulate(self.model):
                        text_inputs = batch["text"]
                        mel_spec = batch["mel"].permute(0, 2, 1)
                        mel_lengths = batch["mel_lengths"]
                        phoneme_mask = batch["phoneme_mask"]  # Get phoneme mask from batch

                        # Pass phoneme_mask to the model
                        loss, cond, pred = self.model(
                            mel_spec,
                            text=text_inputs,
                            lens=mel_lengths,
                            noise_scheduler=self.noise_scheduler,
                            phoneme_mask=phoneme_mask,  # Pass phoneme mask
                        )
                        self.accelerator.backward(loss)

                        if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        if self.is_main:
                            self.ema_model.update()

                        global_update += 1
                        progress_bar.update(1)
                        progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                    if self.accelerator.is_local_main_process:
                        self.accelerator.log(
                            {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                        )
                        if self.logger == "tensorboard":
                            self.writer.add_scalar("loss", loss.item(), global_update)
                            self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                    if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                        self.save_checkpoint(global_update, last=True)

                    if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                        self.save_checkpoint(global_update)

                        if self.log_samples and self.accelerator.is_local_main_process:
                            ref_audio_len = mel_lengths[0]
                            infer_text = [
                                text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                            ]
                            with torch.inference_mode():
                                generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                    cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                    text=infer_text,
                                    duration=ref_audio_len * 2,
                                    steps=nfe_step,
                                    cfg_strength=cfg_strength,
                                    sway_sampling_coef=sway_sampling_coef,
                                )
                                generated = generated.to(torch.float32)
                                gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                                ref_mel_spec = batch["mel"][0].unsqueeze(0)
                                if self.vocoder_name == "vocos":
                                    gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                    ref_audio = vocoder.decode(ref_mel_spec).cpu()
                                elif self.vocoder_name == "bigvgan":
                                    gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                    ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                            torchaudio.save(
                                f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                            )
                            torchaudio.save(
                                f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                            )
                            self.model.train()

            self.save_checkpoint(global_update, last=True)
            self.accelerator.end_training()

    trainer = MaskedPhonemeTrainer(
        model,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project=args.dataset_name,
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id if args.finetune else None,
        log_samples=args.log_samples,
        last_per_updates=args.last_per_updates,
        bnb_optimizer=args.bnb_optimizer,
        mel_spec_type=mel_spec_type,
    )

    print("\nStarting training...")
    trainer.train(
        train_dataset,
        num_workers=args.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
