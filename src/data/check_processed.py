"""Validate the processed GUE promoter JSONL files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PROCESSED_DIR = Path("data/processed/gue_promoter")


def check_split(split: str) -> None:
    path = PROCESSED_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    labels = Counter()
    lengths = []
    examples = []

    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            item = json.loads(line)
            sequence = item["sequence"]
            label = item["label"]

            labels[label] += 1
            lengths.append(len(sequence))
            if idx < 3:
                examples.append(item)

    print("=" * 80)
    print(f"Split: {split}")
    print("=" * 80)
    print(f"Number of examples: {len(lengths)}")
    print(f"Label counts: {dict(labels)}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Mean length: {sum(lengths) / len(lengths):.2f}")
    print()
    print("Examples:")
    for example in examples:
        print(example)
    print()


def main() -> None:
    for split in ["train", "valid", "test"]:
        check_split(split)


if __name__ == "__main__":
    main()
