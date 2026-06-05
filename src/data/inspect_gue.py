"""Inspect raw GUE promoter files and summarize their schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/gue_promoter")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file type: {path}")


def guess_sequence_column(df: pd.DataFrame) -> str:
    candidates = [
        "sequence",
        "seq",
        "dna",
        "promoter",
        "text",
        "text_a",
        "Sequence",
    ]
    for column in candidates:
        if column in df.columns:
            return column

    for column in df.columns:
        if df[column].dtype != object:
            continue
        sample = df[column].dropna().astype(str).head(20)
        if sample.empty:
            continue
        dna_like_ratio = sample.apply(
            lambda value: set(value.upper()).issubset({"A", "C", "G", "T", "N"})
        ).mean()
        if dna_like_ratio > 0.8:
            return column

    raise ValueError("Could not identify a DNA sequence column.")


def guess_label_column(df: pd.DataFrame) -> str:
    candidates = ["label", "labels", "target", "y", "Label"]
    for column in candidates:
        if column in df.columns:
            return column

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique() <= 10:
            return column

    raise ValueError("Could not identify a label column.")


def inspect_file(path: Path) -> None:
    print("=" * 80)
    print(f"File: {path}")
    print("=" * 80)

    df = read_table(path)
    seq_col = guess_sequence_column(df)
    label_col = guess_label_column(df)
    df[seq_col] = df[seq_col].astype(str).str.upper()

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Guessed sequence column: {seq_col}")
    print(f"Guessed label column: {label_col}")
    print()

    lengths = df[seq_col].str.len()
    print("Sequence length stats:")
    print(lengths.describe())
    print()

    print("Label distribution:")
    print(df[label_col].value_counts(dropna=False))
    print()

    invalid = ~df[seq_col].apply(lambda seq: set(seq).issubset({"A", "C", "G", "T", "N"}))
    print(f"Sequences with non-ACGTN characters: {int(invalid.sum())}")
    print(f"Total N bases: {int(df[seq_col].str.count('N').sum())}")
    print()

    print("Example rows:")
    print(df[[seq_col, label_col]].head())
    print()


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"{RAW_DIR} does not exist. Populate it first with download_gue.py."
        )

    files = []
    for pattern in ("*.csv", "*.tsv", "*.txt"):
        files.extend(RAW_DIR.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV/TSV/TXT files found under {RAW_DIR}.")

    for path in sorted(files):
        inspect_file(path)


if __name__ == "__main__":
    main()
