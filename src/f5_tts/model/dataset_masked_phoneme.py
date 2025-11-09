"""
Dataset class for phoneme-based masking training.
Specifically designed for mask-fill fine-tuning on targeted phonemes (e.g., hard 'S' sounds).
"""

import torch
import torchaudio
from datasets import Dataset as Dataset_
from torch.utils.data import Dataset

from f5_tts.model.modules import MelSpec


class MaskedPhonemeDataset(Dataset):
    """
    Dataset that loads audio with phoneme timestamps and creates masks for specific phonemes.

    Expected dataset format (HuggingFace Dataset):
    - 'audio': dict with 'array' and 'sampling_rate'
    - 'text': str
    - 'hard_s_timestamps': list of [start, end] time pairs in seconds (optional, can be any phoneme)
    - 'phoneme_timestamps': list of dicts with 'phoneme', 'start', 'end' (optional, for debugging)
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
        mask_key="hard_s_timestamps",  # key in dataset containing timestamps to mask
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.mask_key = mask_key

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
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def _time_to_frame(self, time_seconds):
        """Convert time in seconds to mel-spectrogram frame index."""
        return int(time_seconds * self.target_sample_rate / self.hop_length)

    def _create_phoneme_mask(self, timestamps, mel_length):
        """
        Create a binary mask for phoneme locations.

        Args:
            timestamps: list of [start, end] time pairs in seconds
            mel_length: total length of mel spectrogram in frames

        Returns:
            mask: boolean tensor of shape [mel_length], True where phoneme should be masked
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
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        # Filter by duration (same as original HFDataset)
        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        # Convert to tensor and resample if needed
        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t'

        # Get mel spectrogram
        mel_spec = self.mel_spectrogram(audio_tensor)
        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        # Get text
        text = row["text"]

        # Create phoneme mask
        phoneme_timestamps = row.get(self.mask_key, None)
        mel_length = mel_spec.shape[-1]
        phoneme_mask = self._create_phoneme_mask(phoneme_timestamps, mel_length)

        return dict(
            mel_spec=mel_spec,
            text=text,
            phoneme_mask=phoneme_mask,  # True where phoneme should be masked
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
