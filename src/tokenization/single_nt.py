"""Single-nucleotide tokenization."""

from __future__ import annotations

from typing import List


def single_nt_tokenize(sequence: str) -> List[str]:
    """Tokenize a DNA sequence into one token per nucleotide."""
    sequence = sequence.upper().strip()
    return list(sequence)


def main() -> None:
    sequence = "ACGTACGT"
    tokens = single_nt_tokenize(sequence)
    print("Sequence:", sequence)
    print("Tokens:", tokens)
    print("Number of tokens:", len(tokens))
    print("Compression ratio:", len(tokens) / len(sequence))


if __name__ == "__main__":
    main()
