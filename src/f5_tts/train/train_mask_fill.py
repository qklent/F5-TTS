"""
Training script for phoneme-based mask filling fine-tuning.
Specifically designed for correcting speech disorders by learning to fill masked phonemes.

Example usage:
python -m f5_tts.train.train_mask_fill --config-name=F5TTS_MaskFill \
    datasets.path=../data/test_processed \
    datasets.mask_key=hard_s_timestamps
"""

import os
import shutil
from importlib.resources import files

import hydra
from cached_path import cached_path
from datasets import load_from_disk, load_dataset
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset_masked_phoneme_optimized_fixed import MemoryOptimizedMaskedPhonemeDataset, collate_fn_masked
from f5_tts.model.utils import get_tokenizer


# Change working directory to root of project (for local editable installs)
os.chdir(str(files("f5_tts").joinpath("../..")))


# Default pretrained checkpoint paths for each model variant
DEFAULT_PRETRAIN_PATHS = {
    "F5TTS_v1_Base": "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "F5TTS_Base": "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt",
    "E2TTS_Base": "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt",
}


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

        # Use custom collate function with memory-optimized settings
        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn_masked,
                num_workers=num_workers,
                pin_memory=False,  # Disable pin_memory to reduce GPU memory usage
                persistent_workers=False,  # Disable persistent workers to prevent memory leaks
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
                pin_memory=False,  # Disable pin_memory to reduce GPU memory usage
                persistent_workers=False,  # Disable persistent workers to prevent memory leaks
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
                        # edit_mask for sample(): True = keep original, False = regenerate
                        sample_phoneme_mask = phoneme_mask[0][:ref_audio_len]
                        edit_mask = (~sample_phoneme_mask).unsqueeze(0)

                        with torch.inference_mode():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=[text_inputs[0]],
                                duration=ref_audio_len,
                                lens=torch.tensor([ref_audio_len], device=mel_spec.device),
                                edit_mask=edit_mask,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32)
                            # Trim to original length and convert to [b, d, t] for vocoder
                            gen_mel_spec = generated[:, :ref_audio_len, :].permute(0, 2, 1).to(self.accelerator.device)
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


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    """Main training function using Hydra configuration."""

    # Extract configuration values
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    # Create experiment name
    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}"

    # Setup checkpoint directory
    checkpoint_path = str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}"))

    # Handle fine-tuning setup
    if model_cfg.ckpts.get("finetune", False):
        # Create checkpoint directory
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

        # Get pretrained checkpoint path
        if model_cfg.ckpts.get("pretrain_path"):
            ckpt_path = model_cfg.ckpts.pretrain_path
        else:
            # Extract base model name (remove _MaskFill suffix)
            base_model_name = model_cfg.model.name.replace("_MaskFill", "")
            if base_model_name in DEFAULT_PRETRAIN_PATHS:
                ckpt_path = str(cached_path(DEFAULT_PRETRAIN_PATHS[base_model_name]))
            else:
                raise ValueError(f"Unknown model variant: {base_model_name}")

        # Copy pretrained checkpoint
        file_checkpoint = os.path.basename(ckpt_path)
        if not file_checkpoint.startswith("pretrained_"):
            file_checkpoint = "pretrained_" + file_checkpoint
        file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
        if not os.path.isfile(file_checkpoint):
            shutil.copy2(ckpt_path, file_checkpoint)
            print(f"Copied pretrained checkpoint to {file_checkpoint}")

    # Setup tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    print(f"\nVocab size: {vocab_size}")
    print(f"Vocoder: {mel_spec_type}")
    print(f"Dataset path: {model_cfg.datasets.path}")
    print(f"Mask key: {model_cfg.datasets.mask_key}")

    # Create model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    # Load datasets
    print("\nLoading datasets...")

    # Check if it's a local path or HuggingFace dataset ID
    dataset_path = model_cfg.datasets.path
    if dataset_path.startswith("../") or dataset_path.startswith("/") or dataset_path.startswith("./"):
        # Local dataset path
        print(f"Loading local dataset from: {dataset_path}")
        hf_dataset = load_from_disk(dataset_path)
    else:
        # HuggingFace dataset ID
        print(f"Loading HuggingFace dataset: {dataset_path}")
        hf_dataset = load_dataset(dataset_path)

    # Check if dataset has train/validation splits
    if hasattr(hf_dataset, "keys"):
        train_hf = hf_dataset["train"]
        print(f"Train samples: {len(train_hf)}")
        if "validation" in hf_dataset.keys():
            print(f"Validation samples: {len(hf_dataset['validation'])}")
    else:
        train_hf = hf_dataset
        print(f"Total samples: {len(train_hf)}")

    # Create masked phoneme dataset
    train_dataset = MemoryOptimizedMaskedPhonemeDataset(
        train_hf,
        mask_key=model_cfg.datasets.mask_key,
        max_audio_length=15.0,  # Reduce max audio length to save memory
        dataset_length=model_cfg.datasets.get('dataset_length', None),
        random_seed=model_cfg.datasets.get('random_seed', None),
        **model_cfg.model.mel_spec,
    )

    print(f"\nDataset loaded successfully with {len(train_dataset)} samples")

    # Create trainer
    trainer = MaskedPhonemeTrainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="F5-TTS-MaskFill",
        wandb_run_name=exp_name,
        wandb_resume_id=None,  # For fine-tuning, we typically don't resume
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    print("\nStarting mask fill training...")
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )

    print("\nMask fill training completed!")


if __name__ == "__main__":
    main()