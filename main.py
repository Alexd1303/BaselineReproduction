import argparse
from pathlib import Path

from Data import CarDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=Path, default="./checkpoints", help="Path to the checkpoints directory")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Path to the dataset directory")
    parser.add_argument("--annotation_dir", type=Path, required=True, help="Path to the annotation directory")

    args = parser.parse_args()

    if not args.annotation_dir.exists():
        print(f"Annotation directory {args.annotation_dir} does not exist. Please provide a valid path.")
        exit(1)

    if not args.dataset_dir.exists():
        print(f"Dataset directory {args.dataset_dir} does not exist. Please provide a valid path.")
        exit(1)

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)