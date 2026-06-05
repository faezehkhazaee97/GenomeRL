"""Download and preprocess an arbitrary GUE task into the shared JSONL format."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.download_gue import download_direct_csvs
from src.data.preprocess_gue import preprocess_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True, help="Local task name, e.g. gue_tf_binding")
    parser.add_argument("--hf-dataset", default="leannmlindsey/GUE")
    parser.add_argument("--hf-config", required=True, help="GUE Hugging Face config name for this task")
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    raw_dir = Path("data/raw") / args.task_name
    processed_dir = Path("data/processed") / args.task_name

    print(f"Downloading {args.hf_dataset}:{args.hf_config} -> {raw_dir}")
    download_direct_csvs(
        dataset_name=args.hf_dataset,
        config_name=args.hf_config,
        out_dir=raw_dir,
    )

    print(f"Preprocessing into {processed_dir}")
    for split in ["train", "valid", "test"]:
        preprocess_split(
            raw_dir=raw_dir,
            out_dir=processed_dir,
            split=split,
            sequence_col=args.sequence_col,
            label_col=args.label_col,
        )

    print("Done.")


if __name__ == "__main__":
    main()
