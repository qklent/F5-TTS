"""
Memory-optimized version of MaskedPhonemeDataset.
Includes explicit memory management and optimizations for large datasets.
"""

import torch
import torchaudio
import gc
import random
from datasets import Dataset as Dataset_
from torch.utils.data import Dataset

from f5_tts.model.modules import MelSpec


class MemoryOptimizedMaskedPhonemeDataset(Dataset):
    """
    Memory-optimized version of MaskedPhonemeDataset for large datasets.
    Includes explicit garbage collection and memory management.
    """

    def __init__(
        self,
        hf_dataset: Dataset_,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        mask_key="hard_s_timestamps",
        max_audio_length=30.0,  # Maximum audio length in seconds
        dataset_length=None,  # Limit dataset to this many samples (None = use full dataset)
        random_seed=42,  # Random seed for reproducible sampling (None = no seed)
    ):
        self.data = hf_dataset
        # Truncate dataset if dataset_length is specified - use random sampling
        if dataset_length is not None and dataset_length < len(hf_dataset):
            # Set random seed for reproducible sampling
            if random_seed is not None:
                random.seed(random_seed)

            # Generate random indices without replacement
            total_samples = len(hf_dataset)
            random_indices = random.sample(range(total_samples), dataset_length)
            self.data = hf_dataset.select(random_indices)

        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.mask_key = mask_key
        self.max_audio_length = max_audio_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        """Get the length of mel spectrogram in frames for a given sample."""
        try:
            row = self.data[index]
            audio = row["audio"]["array"]
            sample_rate = row["audio"]["sampling_rate"]
            return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length
        except Exception:
            # Return a reasonable default if there's an error
            return 1000  # ~10 seconds at 100 fps

    def __len__(self):
        return len(self.data)

    def _time_to_frame(self, time_seconds):
        """Convert time in seconds to mel-spectrogram frame index."""
        return int(time_seconds * self.target_sample_rate / self.hop_length)

    def _create_phoneme_mask(self, timestamps, mel_length):
        """
        Create a binary mask for phoneme locations.
        """
        mask = torch.zeros(mel_length, dtype=torch.bool)

        if timestamps is None or len(timestamps) == 0:
            return mask

        for start_time, end_time in timestamps:
            start_frame = self._time_to_frame(start_time)
            end_frame = self._time_to_frame(end_time)

            # Clamp to valid range
            start_frame = max(0, min(start_frame, mel_length - 1))
            end_frame = max(0, min(end_frame, mel_length))

            if start_frame < end_frame:
                mask[start_frame:end_frame] = True

        return mask

    def __getitem__(self, index):
        max_retries = 5
        for retry in range(max_retries):
            try:
                print(f"[DEBUG] Starting __getitem__ for index {index}, retry {retry}")

                # Memory info before starting
                if torch.cuda.is_available():
                    print(f"[DEBUG] GPU memory before: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated, {torch.cuda.memory_reserved() / 1024**2:.1f}MB reserved")

                print(f"[DEBUG] Loading data row for index {index}")
                row = self.data[index]
                print(f"[DEBUG] Successfully loaded data row")

                print(f"[DEBUG] Extracting audio array")
                audio = row["audio"]["array"]
                sample_rate = row["audio"]["sampling_rate"]
                duration = audio.shape[-1] / sample_rate
                print(f"[DEBUG] Audio extracted - shape: {audio.shape}, sample_rate: {sample_rate}, duration: {duration:.2f}s")

                # Filter by duration
                if duration > self.max_audio_length or duration < 0.3:
                    print(f"[DEBUG] Skipping audio due to duration ({duration:.2f}s)")
                    index = (index + 1) % len(self.data)
                    continue

                # Convert to tensor (with explicit dtype to save memory)
                print(f"[DEBUG] Converting numpy array to torch tensor - size: {audio.nbytes / 1024**2:.1f}MB")
                audio_tensor = torch.from_numpy(audio).float()
                print(f"[DEBUG] Audio tensor created - shape: {audio_tensor.shape}")

                # Resample if needed
                if sample_rate != self.target_sample_rate:
                    print(f"[DEBUG] Resampling from {sample_rate} to {self.target_sample_rate}")
                    resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                    print(f"[DEBUG] Resampler created, applying transformation")
                    audio_tensor = resampler(audio_tensor)
                    print(f"[DEBUG] Resampling complete - new shape: {audio_tensor.shape}")
                    # Explicit cleanup of resampler
                    del resampler
                    print(f"[DEBUG] Resampler deleted")

                print(f"[DEBUG] Adding batch dimension")
                audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t'
                print(f"[DEBUG] Audio tensor final shape: {audio_tensor.shape}")

                # Memory info after audio processing
                if torch.cuda.is_available():
                    print(f"[DEBUG] GPU memory after audio processing: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated")

                # Get mel spectrogram
                try:
                    print(f"[DEBUG] Computing mel spectrogram")
                    mel_spec = self.mel_spectrogram(audio_tensor)
                    print(f"[DEBUG] Mel spectrogram computed - shape: {mel_spec.shape}")
                    mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'
                    print(f"[DEBUG] Mel spectrogram squeezed - final shape: {mel_spec.shape}")
                except Exception as e:
                    print(f"[ERROR] Error processing mel spectrogram for sample {index}: {e}")
                    index = (index + 1) % len(self.data)
                    continue

                # Clean up audio tensor to free memory
                print(f"[DEBUG] Cleaning up audio tensor")
                del audio_tensor
                print(f"[DEBUG] Audio tensor deleted")

                # Memory info after mel processing
                if torch.cuda.is_available():
                    print(f"[DEBUG] GPU memory after mel processing: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated")

                # Get text
                print(f"[DEBUG] Extracting text")
                text = row["text"]
                print(f"[DEBUG] Text extracted - length: {len(text)} characters")

                # Create phoneme mask
                print(f"[DEBUG] Creating phoneme mask")
                phoneme_timestamps = row.get(self.mask_key, None)
                mel_length = mel_spec.shape[-1]
                print(f"[DEBUG] Phoneme timestamps: {len(phoneme_timestamps) if phoneme_timestamps else 0} segments, mel_length: {mel_length}")
                phoneme_mask = self._create_phoneme_mask(phoneme_timestamps, mel_length)
                print(f"[DEBUG] Phoneme mask created - shape: {phoneme_mask.shape}")

                # Explicit garbage collection every 100 samples
                if index % 100 == 0:
                    print(f"[DEBUG] Running garbage collection at index {index}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print(f"[DEBUG] GPU cache cleared")

                print(f"[DEBUG] Successfully completed __getitem__ for index {index}")

                # Final memory info
                if torch.cuda.is_available():
                    print(f"[DEBUG] Final GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated")

                return dict(
                    mel_spec=mel_spec,
                    text=text,
                    phoneme_mask=phoneme_mask,
                )

            except Exception as e:
                print(f"Error loading sample {index}: {e}")
                index = (index + 1) % len(self.data)
                continue

        # If we've tried max_retries times, return a dummy sample
        print(f"Failed to load any valid samples after {max_retries} retries, returning dummy sample")
        dummy_mel = torch.zeros((100, 100))  # 100 mel channels, 100 frames
        dummy_mask = torch.zeros(100, dtype=torch.bool)
        return dict(
            mel_spec=dummy_mel,
            text="",
            phoneme_mask=dummy_mask,
        )


def collate_fn_masked(batch):
    """
    Collate function for MaskedPhonemeDataset.
    Handles padding of mel spectrograms, texts, and phoneme masks.
    """
    mel_specs = [item["mel_spec"] for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    # Pad mel spectrograms
    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = torch.nn.functional.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)
    mel_specs = torch.stack(padded_mel_specs)

    # Pad phoneme masks
    phoneme_masks = [item["phoneme_mask"] for item in batch]
    padded_masks = []
    for mask in phoneme_masks:
        padding = (0, max_mel_length - mask.size(0))
        padded_mask = torch.nn.functional.pad(mask, padding, value=False)
        padded_masks.append(padded_mask)
    phoneme_masks = torch.stack(padded_masks)

    # Text
    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
        phoneme_mask=phoneme_masks,  # [batch, mel_length], True where to mask
    )
