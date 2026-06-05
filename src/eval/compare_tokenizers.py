"""Compare baseline tokenizers on processed GUE promoter sequences."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from src.tokenization.boundary_utils import average_token_length, compression_ratio
from src.tokenization.fixed_kmer import fixed_kmer_tokenize
from src.tokenization.random_policy import random_tokenize
from src.tokenization.single_nt import single_nt_tokenize


DATA_PATH = Path("data/processed/gue_promoter/train.jsonl")


def load_sequences(path: Path, max_examples: int = 1000):
    sequences = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= max_examples:
                break
            item = json.loads(line)
            sequences.append(item["sequence"])
    return sequences


def summarize_tokenizer(name, tokenize_fn, sequences):
    num_tokens = []
    compression_ratios = []
    avg_token_lengths = []

    for sequence in sequences:
        tokens = tokenize_fn(sequence)
        num_tokens.append(len(tokens))
        compression_ratios.append(compression_ratio(sequence, tokens))
        avg_token_lengths.append(average_token_length(tokens))

    result = {
        "name": name,
        "examples": len(sequences),
        "avg_num_tokens": mean(num_tokens),
        "avg_compression_ratio": mean(compression_ratios),
        "avg_token_length": mean(avg_token_lengths),
    }

    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"Examples: {result['examples']}")
    print(f"Avg number of tokens: {result['avg_num_tokens']:.2f}")
    print(f"Avg compression ratio: {result['avg_compression_ratio']:.4f}")
    print(f"Avg token length: {result['avg_token_length']:.2f}")
    print()

    return result


def main() -> None:
    sequences = load_sequences(DATA_PATH, max_examples=1000)
    results = []

    results.append(summarize_tokenizer("Single nucleotide", single_nt_tokenize, sequences))
    results.append(
        summarize_tokenizer(
            "Fixed 6-mer non-overlapping",
            lambda seq: fixed_kmer_tokenize(seq, k=6, overlap=False),
            sequences,
        )
    )
    results.append(
        summarize_tokenizer(
            "Fixed 6-mer overlapping",
            lambda seq: fixed_kmer_tokenize(seq, k=6, overlap=True),
            sequences,
        )
    )
    results.append(
        summarize_tokenizer(
            "Random boundary p=0.2",
            lambda seq: random_tokenize(seq, boundary_prob=0.2, seed=42),
            sequences,
        )
    )
    results.append(
        summarize_tokenizer(
            "Random boundary p=0.5",
            lambda seq: random_tokenize(seq, boundary_prob=0.5, seed=42),
            sequences,
        )
    )

    try:
        from src.tokenization.dnabert2_bpe import DNABERT2BPETokenizer

        bpe = DNABERT2BPETokenizer()
        results.append(summarize_tokenizer("DNABERT-2 BPE", bpe.tokenize, sequences))
    except Exception as exc:  # pragma: no cover - depends on model download access
        print("=" * 80)
        print("DNABERT-2 BPE")
        print("=" * 80)
        print("Skipped because tokenizer could not be loaded.")
        print(f"Error: {exc}")
        print()

    out_path = Path("results/baselines/tokenizer_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
