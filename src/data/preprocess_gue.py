"""Preprocess raw GUE promoter files into a shared JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/gue_promoter")
OUT_DIR = Path("data/processed/gue_promoter")

DEFAULT_SEQUENCE_COL = "sequence"
DEFAULT_LABEL_COL = "label"
SPLIT_FILE_CANDIDATES = {
    "train": ["train.csv", "train.tsv", "train.txt"],
    "valid": ["valid.csv", "valid.tsv", "dev.csv", "dev.tsv", "validation.csv", "validation.tsv"],
    "test": ["test.csv", "test.tsv", "test.txt"],
}


def find_split_file(raw_dir: Path, split: str) -> Path:
    for name in SPLIT_FILE_CANDIDATES[split]:
        path = raw_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find a file for split '{split}' under {raw_dir}. "
        f"Tried: {SPLIT_FILE_CANDIDATES[split]}"
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file type: {path}")


def clean_sequence(sequence: str) -> str:
    return str(sequence).upper().strip()


def is_valid_sequence(sequence: str) -> bool:
    return set(sequence).issubset({"A", "C", "G", "T"})


def preprocess_split(
    raw_dir: Path,
    out_dir: Path,
    split: str,
    sequence_col: str,
    label_col: str,
) -> None:
    path = find_split_file(raw_dir, split)
    df = read_table(path)

    if sequence_col not in df.columns:
        raise ValueError(
            f"Sequence column '{sequence_col}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    rows = []
    for idx, row in df.iterrows():
        sequence = clean_sequence(row[sequence_col])
        if not sequence or not is_valid_sequence(sequence):
            continue

        rows.append(
            {
                "id": f"{split}_{idx:06d}",
                "sequence": sequence,
                "label": int(row[label_col]),
                "split": split,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item) + "\n")

    print(f"Processed {split}: {len(rows)} examples")
    print(f"Saved to: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--sequence-col", default=DEFAULT_SEQUENCE_COL)
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    for split in ["train", "valid", "test"]:
        preprocess_split(
            raw_dir=raw_dir,
            out_dir=out_dir,
            split=split,
            sequence_col=args.sequence_col,
            label_col=args.label_col,
        )


if __name__ == "__main__":
    main()
