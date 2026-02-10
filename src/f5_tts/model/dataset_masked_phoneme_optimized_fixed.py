"""
Memory-optimized version of MaskedPhonemeDataset.
Includes explicit memory management and optimizations for large datasets.
Fixed version that avoids TorchCodec issues by handling audio loading more robustly.
"""

import torch
import torchaudio
import gc
import random
import numpy as np
from datasets import Dataset as Dataset_
from torch.utils.data import Dataset

from f5_tts.model.modules import MelSpec


class MemoryOptimizedMaskedPhonemeDataset(Dataset):
    """
    Memory-optimized version of MaskedPhonemeDataset for large datasets.
    Includes explicit garbage collection and memory management.
    Fixed to handle TorchCodec issues gracefully.
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
        mask_margin_ms=30.0,  # Margin in milliseconds to extend mask around phoneme boundaries
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
        self.mask_margin_s = mask_margin_ms / 1000.0  # convert to seconds
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
            # Safe audio loading
            audio, sample_rate = self._safe_load_audio(row)
            if audio is None:
                return 1000  # Default fallback
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
            start_frame = self._time_to_frame(start_time - self.mask_margin_s)
            end_frame = self._time_to_frame(end_time + self.mask_margin_s)

            # Clamp to valid range
            start_frame = max(0, min(start_frame, mel_length - 1))
            end_frame = max(0, min(end_frame, mel_length))

            if start_frame < end_frame:
                mask[start_frame:end_frame] = True

        return mask

    def _safe_load_audio(self, row):
        """
        Safely load audio from dataset row, handling TorchCodec failures.
        Returns (audio_array, sample_rate) or (None, None) if failed.
        """
        # Debug: print available keys in the row
        # print(f"[DEBUG] Available keys in dataset row: {list(row.keys())}")

        # # Try alternative approaches first since torchcodec AudioDecoder is not available
        # try:
        #     # If the dataset has a 'path' or 'file' field, try loading directly first
        #     if "path" in row:
        #         audio_path = row["path"]
        #         print(f"[DEBUG] Trying to load audio from path: {audio_path}")
        #         audio_tensor, sample_rate = torchaudio.load(audio_path)
        #         return audio_tensor.numpy().squeeze(), sample_rate
        #     elif "file" in row:
        #         # If it's a file-like object, try to get path
        #         audio_path = row["file"]
        #         print(f"[DEBUG] Trying to load audio from file: {audio_path}")
        #         if hasattr(audio_path, 'name'):
        #             audio_tensor, sample_rate = torchaudio.load(audio_path.name)
        #             return audio_tensor.numpy().squeeze(), sample_rate
        #         elif isinstance(audio_path, str):
        #             audio_tensor, sample_rate = torchaudio.load(audio_path)
        #             return audio_tensor.numpy().squeeze(), sample_rate
        #     elif "audio_path" in row:
        #         audio_path = row["audio_path"]
        #         print(f"[DEBUG] Trying to load audio from audio_path: {audio_path}")
        #         audio_tensor, sample_rate = torchaudio.load(audio_path)
        #         return audio_tensor.numpy().squeeze(), sample_rate
        # except Exception as file_loading_error:
        #     print(f"[WARNING] File-based audio loading failed: {str(file_loading_error)[:100]}...")

        # Fall back to trying torchcodec (will likely fail but worth trying)
        try:
            # print("[DEBUG] Trying torchcodec audio loading...")
            # First try to access the audio normally
            audio = row["audio"]["array"]
            sample_rate = row["audio"]["sampling_rate"]
            return audio, sample_rate
        except Exception as torchcodec_error:
            print(f"[WARNING] TorchCodec failed: {str(torchcodec_error)[:100]}...")

            # Try to see if audio data is stored as bytes
            try:
                audio_data = row.get("audio", {})
                # print(
                #     f"[DEBUG] Audio data type: {type(audio_data)}, keys: {list(audio_data.keys()) if isinstance(audio_data, dict) else 'N/A'}"
                # )
                if isinstance(audio_data, dict) and "bytes" in audio_data:
                    # Handle bytes data (would need more specific implementation)
                    # print("[WARNING] Audio stored as bytes - not implemented yet")
                    return None, None
            except Exception as fallback_error:
                print(f"[WARNING] Fallback audio loading also failed: {str(fallback_error)[:100]}...")

            return None, None

    def __getitem__(self, index):
        max_retries = 5
        for retry in range(max_retries):
            try:
                # print(f"[DEBUG] Starting __getitem__ for index {index}, retry {retry}")

                # # Memory info before starting
                # if torch.cuda.is_available():
                #     print(
                #         f"[DEBUG] GPU memory before: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated, {torch.cuda.memory_reserved() / 1024**2:.1f}MB reserved"
                #     )

                # print(f"[DEBUG] Loading data row for index {index}")
                row = self.data[index]
                # print("[DEBUG] Successfully loaded data row")

                # print("[DEBUG] Extracting audio array (safe mode)")
                # Use safe audio loading
                audio, sample_rate = self._safe_load_audio(row)

                if audio is None:
                    # print(f"[WARNING] Could not load audio for sample {index}, skipping")
                    index = (index + 1) % len(self.data)
                    continue

                duration = audio.shape[-1] / sample_rate
                # print(
                #     f"[DEBUG] Audio extracted - shape: {audio.shape}, sample_rate: {sample_rate}, duration: {duration:.2f}s"
                # )

                # Filter by duration
                if duration > self.max_audio_length or duration < 0.3:
                    # print(f"[DEBUG] Skipping audio due to duration ({duration:.2f}s)")
                    index = (index + 1) % len(self.data)
                    continue

                # Convert to tensor (with explicit dtype to save memory)
                # print(f"[DEBUG] Converting array to torch tensor - size: {audio.nbytes / 1024**2:.1f}MB")
                if isinstance(audio, np.ndarray):
                    audio_tensor = torch.from_numpy(audio).float()
                else:
                    audio_tensor = torch.tensor(audio).float()
                # print(f"[DEBUG] Audio tensor created - shape: {audio_tensor.shape}")

                # Resample if needed
                if sample_rate != self.target_sample_rate:
                    # print(f"[DEBUG] Resampling from {sample_rate} to {self.target_sample_rate}")
                    resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                    # print("[DEBUG] Resampler created, applying transformation")
                    audio_tensor = resampler(audio_tensor)
                    # print(f"[DEBUG] Resampling complete - new shape: {audio_tensor.shape}")
                    # Explicit cleanup of resampler
                    del resampler
                    # print("[DEBUG] Resampler deleted")

                # print("[DEBUG] Adding batch dimension")
                audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t'
                # print(f"[DEBUG] Audio tensor final shape: {audio_tensor.shape}")

                # # Memory info after audio processing
                # if torch.cuda.is_available():
                #     print(
                #         f"[DEBUG] GPU memory after audio processing: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated"
                #     )

                # Get mel spectrogram
                try:
                    # print("[DEBUG] Computing mel spectrogram")
                    mel_spec = self.mel_spectrogram(audio_tensor)
                    # print(f"[DEBUG] Mel spectrogram computed - shape: {mel_spec.shape}")
                    mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'
                    # print(f"[DEBUG] Mel spectrogram squeezed - final shape: {mel_spec.shape}")
                except Exception as e:
                    print(f"[ERROR] Error processing mel spectrogram for sample {index}: {e}")
                    index = (index + 1) % len(self.data)
                    continue

                # Clean up audio tensor to free memory
                # print("[DEBUG] Cleaning up audio tensor")
                del audio_tensor
                # print("[DEBUG] Audio tensor deleted")

                # Memory info after mel processing
                # if torch.cuda.is_available():
                #     print(
                #         f"[DEBUG] GPU memory after mel processing: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated"
                #     )

                # Get text
                # print("[DEBUG] Extracting text")
                text = row["text"]
                # print(f"[DEBUG] Text extracted - length: {len(text)} characters")

                # Create phoneme mask
                # print("[DEBUG] Creating phoneme mask")
                phoneme_timestamps = row.get(self.mask_key, None)
                mel_length = mel_spec.shape[-1]
                # print(
                #     f"[DEBUG] Phoneme timestamps: {len(phoneme_timestamps) if phoneme_timestamps else 0} segments, mel_length: {mel_length}"
                # )
                phoneme_mask = self._create_phoneme_mask(phoneme_timestamps, mel_length)
                # print(f"[DEBUG] Phoneme mask created - shape: {phoneme_mask.shape}")

                # Explicit garbage collection every 100 samples
                if index % 100 == 0:
                    # print(f"[DEBUG] Running garbage collection at index {index}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # print("[DEBUG] GPU cache cleared")

                # print(f"[DEBUG] Successfully completed __getitem__ for index {index}")

                # # Final memory info
                # if torch.cuda.is_available():
                #     print(f"[DEBUG] Final GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated")

                return {
                    "mel": mel_spec,
                    "text": text,
                    "phoneme_mask": phoneme_mask,
                    "mel_lengths": mel_length,
                }

            except Exception as e:
                print(f"Error loading sample {index}: {e}")
                index = (index + 1) % len(self.data)
                continue

        # If we've tried max_retries times, return a dummy sample
        print(f"Failed to load any valid samples after {max_retries} retries, returning dummy sample")
        dummy_mel = torch.zeros((100, 100))  # 100 mel channels, 100 frames
        dummy_mask = torch.ones(100, dtype=torch.bool)  # all-True to avoid NaN loss from empty mask
        return {
            "mel": dummy_mel,
            "text": "dummy text sample",  # Non-empty text to avoid tokenization issues
            "phoneme_mask": dummy_mask,
            "mel_lengths": 100,
        }


def collate_fn_masked(batch):
    """
    Collate function for MaskedPhonemeDataset.
    Handles padding of mel spectrograms, texts, and phoneme masks.
    """
    mel_specs = [item["mel"] for item in batch]
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
