#!/usr/bin/env python3
"""
Dataset Filtering Script

This script:
1. Loads the dataset from local checkpoint
2. Filters to keep only examples containing hard 'S' (s̪) sounds
3. Displays filtering statistics
4. Uploads the filtered dataset to HuggingFace Hub as a private repository

Usage:
    python filter_dataset.py --dataset-path <path> --repo-name <repo_name>

Remember to login to HuggingFace first:
    huggingface-cli login
"""

import argparse
import os
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk
from dotenv import load_dotenv


def load_chunked_dataset(base_path, split="train"):
    """Load all chunks from a chunked dataset and concatenate them."""
    split_path = Path(base_path) / split
    chunk_dirs = sorted([d for d in split_path.iterdir() if d.is_dir() and d.name.startswith("chunk_")])

    print(f"Found {len(chunk_dirs)} chunks for {split} split")

    datasets = []
    for chunk_dir in chunk_dirs:
        print(f"Loading {chunk_dir.name}...")
        chunk_dataset = load_from_disk(str(chunk_dir))
        datasets.append(chunk_dataset)

    # Concatenate all chunks
    combined = concatenate_datasets(datasets)
    print(f"Combined {split} dataset: {len(combined)} examples")
    return combined


def has_hard_s(phoneme_timestamps):
    """
    Check if the phoneme_timestamps contains hard 'S' sound (s̪ phoneme).
    Returns True if at least one hard 'S' is present, False otherwise.

    Args:
        phoneme_timestamps: List of phoneme info dicts or None
    """
    if phoneme_timestamps is None:
        return False

    for phoneme_info in phoneme_timestamps:
        # Hard 'S' is represented as 's̪' in IPA (with diacritic)
        if phoneme_info["phoneme"] == "s̪":
            return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Filter dataset for hard 'S' sounds")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/qklent/programming/speech_disorder_correction/mlm/data_preparation/data/prod_checkpoint",
        help="Path to the dataset checkpoint"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="qklent/tonebooks-mfa-phonemes-only-hard-s",
        help="HuggingFace repository name (format: username/repo-name)"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading to HuggingFace Hub"
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of processes for filtering"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(".env")

    # Load train and validation splits
    print("Loading train split...")
    train_dataset = load_chunked_dataset(args.dataset_path, "train")

    print("\nLoading validation split...")
    val_dataset = load_chunked_dataset(args.dataset_path, "validation")

    # Create DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    print(f"\nOriginal dataset: {dataset}")
    print(f"Total examples - Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}")

    # Filter dataset to keep only examples with hard 'S'
    print("\nFiltering dataset to keep only examples with hard 'S' (s̪)...")

    # Filter by only passing the phoneme_timestamps column to avoid decoding audio
    # This prevents errors from corrupted audio files
    filtered_dataset = dataset.filter(
        has_hard_s,
        input_columns=["phoneme_timestamps"],
        load_from_cache_file=False,
        num_proc=args.num_proc,
        desc="Filtering for hard 'S' sounds"
    )

    print(f"\nFiltered dataset: {filtered_dataset}")
    print(f"Filtered examples - Train: {len(filtered_dataset['train'])}, Validation: {len(filtered_dataset['validation'])}")

    # Display statistics
    original_train = len(dataset["train"])
    original_val = len(dataset["validation"])
    filtered_train = len(filtered_dataset["train"])
    filtered_val = len(filtered_dataset["validation"])

    print("\n=== Filtering Statistics ===")
    print(f"Train set: {filtered_train}/{original_train} ({filtered_train / original_train * 100:.1f}% kept)")
    print(f"Validation set: {filtered_val}/{original_val} ({filtered_val / original_val * 100:.1f}% kept)")
    print(
        f"Total: {filtered_train + filtered_val}/{original_train + original_val} "
        f"({(filtered_train + filtered_val) / (original_train + original_val) * 100:.1f}% kept)"
    )

    # Verify an example from filtered dataset
    if len(filtered_dataset["train"]) > 0:
        example = filtered_dataset["train"][0]
        hard_s_count = sum(1 for p in example["phoneme_timestamps"] if p["phoneme"] == "s̪")
        print(f"\nExample text: {example['text']}")
        print(f"Number of hard 'S' sounds: {hard_s_count}")
        print(f"Voice: {example['voice_name']}")

    # Push to HuggingFace Hub (private)
    if not args.no_upload:
        print(f"\nPushing filtered dataset to HuggingFace Hub: {args.repo_name}")
        print("Note: This will be a private dataset")

        # Disable progress bars to avoid context issues
        import datasets
        datasets.disable_progress_bar()

        filtered_dataset.push_to_hub(
            args.repo_name,
            private=True,
            token=os.getenv("HF_TOKEN"),
        )

        # Re-enable progress bars
        datasets.enable_progress_bar()

        print(f"\nDataset successfully uploaded to: https://huggingface.co/datasets/{args.repo_name}")
    else:
        print("\nSkipping upload to HuggingFace Hub (--no-upload flag set)")


if __name__ == "__main__":
    main()
