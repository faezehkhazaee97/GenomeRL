"""Download helpers for the GUE promoter dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - depends on local environment
    load_dataset = None


RAW_DIR = Path("data/raw/gue_promoter")
DEFAULT_HF_DATASET = "leannmlindsey/GUE"
# The proposal describes ~300 bp promoter windows, so use the 300 bp subset first.
DEFAULT_HF_CONFIG = "prom_300_all"
HF_RESOLVE_PREFIX = "https://huggingface.co/datasets/{dataset}/resolve/main/GUE/{config}"


def _normalize_split_name(split_name: str) -> str:
    lowered = split_name.lower()
    if lowered == "train":
        return "train"
    if lowered in {"validation", "valid", "dev"}:
        return "dev"
    if lowered == "test":
        return "test"
    raise ValueError(f"Unsupported split name from dataset source: {split_name}")


def download_from_huggingface(dataset_name: str, config_name: str, out_dir: Path) -> None:
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' package is not installed. Install requirements first."
        )

    print(f"Loading dataset '{dataset_name}' with config '{config_name}'...")
    dataset = load_dataset(dataset_name, name=config_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_dataset in dataset.items():
        normalized = _normalize_split_name(split_name)
        out_path = out_dir / f"{normalized}.csv"
        split_dataset.to_csv(str(out_path), index=False)
        print(f"Saved {split_name} -> {out_path}")

    print()
    print("Done. Expected next step:")
    print("  python -m src.data.inspect_gue")


def download_direct_csvs(dataset_name: str, config_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = HF_RESOLVE_PREFIX.format(dataset=dataset_name, config=config_name)

    for split_name in ("train", "dev", "test"):
        url = f"{base_url}/{split_name}.csv"
        out_path = out_dir / f"{split_name}.csv"
        print(f"Downloading {url}")
        urlretrieve(url, out_path)
        print(f"Saved to {out_path}")

    print()
    print("Done. Expected next step:")
    print("  python -m src.data.inspect_gue")


def print_manual_layout(out_dir: Path) -> None:
    print("Manual-download mode selected.")
    print(f"Place the raw files under: {out_dir}")
    print()
    print("Expected filenames:")
    print("  train.csv")
    print("  dev.csv or valid.csv")
    print("  test.csv")
    print()
    print("If your source provides separate subsets, a layout like this is also fine:")
    print("  data/raw/gue_promoter/prom_300_tata/train.csv")
    print("  data/raw/gue_promoter/prom_300_notata/train.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["direct", "huggingface", "manual"],
        default="direct",
        help="How to populate data/raw/gue_promoter.",
    )
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        help="Hugging Face dataset name to use when --source=huggingface.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_HF_CONFIG,
        help="Hugging Face dataset config to use when --source=huggingface.",
    )
    parser.add_argument(
        "--out",
        default=str(RAW_DIR),
        help="Output directory for raw dataset files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out)

    if args.source == "manual":
        out_dir.mkdir(parents=True, exist_ok=True)
        print_manual_layout(out_dir)
        return

    if args.source == "direct":
        download_direct_csvs(
            dataset_name=args.hf_dataset,
            config_name=args.config,
            out_dir=out_dir,
        )
        return

    download_from_huggingface(
        dataset_name=args.hf_dataset,
        config_name=args.config,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
