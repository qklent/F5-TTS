#!/usr/bin/env python3
"""
Script to add hard_s_timestamps field to the dataset.
This extracts timestamps for 's̪' phonemes from the existing phoneme_timestamps field.
"""

from datasets import load_dataset

def extract_hard_s_timestamps(example):
    """
    Extract timestamps for hard 'S' (s̪) phonemes from phoneme_timestamps.
    Adds a new field 'hard_s_timestamps' with list of [start, end] pairs.
    """
    phoneme_timestamps = example.get('phoneme_timestamps', [])
    hard_s_timestamps = []

    if phoneme_timestamps:
        for phoneme_info in phoneme_timestamps:
            # Check for hard 'S' phoneme (s̪)
            if phoneme_info.get("phoneme") == "s̪":
                start_time = phoneme_info.get("start", 0.0)
                end_time = phoneme_info.get("end", 0.0)
                hard_s_timestamps.append([start_time, end_time])

    # Add the new field to the example
    example['hard_s_timestamps'] = hard_s_timestamps
    return example

def main():
    # Load your dataset
    print("Loading dataset...")
    dataset = load_dataset('qklent/tonebooks-mfa-phonemes-only-hard-s')

    # Process both splits
    print("Processing train split...")
    train_processed = dataset['train'].map(
        extract_hard_s_timestamps,
        desc="Adding hard_s_timestamps to train split"
    )

    print("Processing validation split...")
    val_processed = dataset['validation'].map(
        extract_hard_s_timestamps,
        desc="Adding hard_s_timestamps to validation split"
    )

    # Check results
    print("\n=== Results ===")

    # Count samples with hard S
    train_with_hard_s = sum(1 for ex in train_processed if len(ex['hard_s_timestamps']) > 0)
    val_with_hard_s = sum(1 for ex in val_processed if len(ex['hard_s_timestamps']) > 0)

    print(f"Train samples with hard 'S': {train_with_hard_s}/{len(train_processed)}")
    print(f"Validation samples with hard 'S': {val_with_hard_s}/{len(val_processed)}")

    # Show example
    if train_with_hard_s > 0:
        # Find first example with hard S
        for ex in train_processed:
            if len(ex['hard_s_timestamps']) > 0:
                print(f"\nExample text: {ex['text']}")
                print(f"Hard S timestamps: {ex['hard_s_timestamps']}")
                break

    # Save updated dataset
    print("\nSaving updated dataset...")
    from datasets import DatasetDict

    updated_dataset = DatasetDict({
        'train': train_processed,
        'validation': val_processed
    })

    # Save locally first (optional)
    updated_dataset.save_to_disk("./dataset_with_hard_s")

    # Push updated dataset back to HuggingFace
    print("Pushing updated dataset to HuggingFace...")
    updated_dataset.push_to_hub(
        'qklent/tonebooks-mfa-phonemes-only-hard-s',
        private=True
    )

    print("Dataset updated successfully!")

if __name__ == "__main__":
    main()