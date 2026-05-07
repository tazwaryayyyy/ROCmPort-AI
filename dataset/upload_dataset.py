"""
Upload cuda-to-rocm-wavefront-bugs dataset to HuggingFace Hub.

Supports either:
  - 17 individual batch JSON files (recommended)
  - A single combined JSONL file

Usage (individual files in current dir):
    python upload_dataset.py --token hf_xxxx --files_dir .

Usage (specific directory):
    python upload_dataset.py --token hf_xxxx --files_dir ./my_batches/

Usage (single JSONL fallback):
    python upload_dataset.py --token hf_xxxx --jsonl cuda_rocm_bugs.jsonl
"""
import json
import os
import glob
import argparse
from collections import Counter
from datasets import Dataset, DatasetDict, Features, Value

FEATURES = Features({
    "id": Value("string"),
    "bug_category": Value("string"),
    "risk_level": Value("string"),
    "kernel_type": Value("string"),
    "cuda_snippet": Value("string"),
    "hip_naive": Value("string"),
    "hip_corrected": Value("string"),
    "explanation": Value("string"),
    "amd_hardware": Value("string"),
    "rocm_version": Value("string"),
    "verified_on_mi300x": Value("bool"),
    "hipify_catches_this": Value("bool"),
})


def load_from_files(files_dir):
    """Load all batch JSON files from a directory."""
    pattern = os.path.join(files_dir, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .json files found in {files_dir}")

    all_examples = []
    for f in files:
        with open(f) as fp:
            batch = json.load(fp)
            all_examples.extend(batch)
        print(f"  Loaded {len(batch):>3} examples from {os.path.basename(f)}")

    return all_examples


def load_from_jsonl(jsonl_path):
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True,
                        help="HuggingFace write token")
    parser.add_argument(
        "--repo", default="tazwarrrr/cuda-to-rocm-wavefront-bugs")
    parser.add_argument("--files_dir", default=None,
                        help="Directory with batch .json files")
    parser.add_argument("--jsonl", default=None,
                        help="Single combined JSONL file")
    args = parser.parse_args()

    # Auto-detect if nothing specified
    if not args.files_dir and not args.jsonl:
        json_files = sorted(glob.glob("batch_*.json"))
        if json_files:
            print(
                f"Auto-detected {len(json_files)} batch files in current directory")
            args.files_dir = "."
        elif os.path.exists("cuda_rocm_bugs.jsonl"):
            args.jsonl = "cuda_rocm_bugs.jsonl"
        else:
            parser.error("Provide --files_dir or --jsonl")

    # Load data
    print("\nLoading data...")
    if args.files_dir:
        data = load_from_files(args.files_dir)
    else:
        data = load_from_jsonl(args.jsonl)

    print(f"\nTotal: {len(data)} examples")
    print("\nBy category:")
    for cat, count in sorted(Counter(e["bug_category"] for e in data).items()):
        print(f"  {cat}: {count}")

    # Build dataset with 90/10 split
    ds = Dataset.from_list(data, features=FEATURES)
    split = ds.train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict(
        {"train": split["train"], "test": split["test"]})

    print(
        f"\nSplit: {len(dataset_dict['train'])} train / {len(dataset_dict['test'])} test")
    print(f"\nUploading to https://huggingface.co/datasets/{args.repo} ...")

    dataset_dict.push_to_hub(args.repo, token=args.token, private=False)

    print("\n✅ Done!")
    print(f"   https://huggingface.co/datasets/{args.repo}")
    print("\nNext steps:")
    print("  1. Paste dataset_README.md as the Dataset Card on HuggingFace")
    print("  2. Link dataset in lablab.ai submission under 'HuggingFace' track")


if __name__ == "__main__":
    main()
