"""Fixed k-mer tokenization."""

from __future__ import annotations

from typing import List


def fixed_kmer_tokenize(
    sequence: str,
    k: int = 6,
    overlap: bool = False,
    drop_remainder: bool = False,
) -> List[str]:
    """Tokenize a DNA sequence into fixed-length k-mers."""
    sequence = sequence.upper().strip()
    if k <= 0:
        raise ValueError("k must be positive.")
    if not sequence:
        return []

    tokens = []
    if overlap:
        for index in range(0, len(sequence) - k + 1):
            tokens.append(sequence[index : index + k])
    else:
        for index in range(0, len(sequence), k):
            token = sequence[index : index + k]
            if drop_remainder and len(token) < k:
                continue
            tokens.append(token)
    return tokens


def main() -> None:
    sequence = "ACGTACGT"
    print("Sequence:", sequence)

    tokens = fixed_kmer_tokenize(sequence, k=6, overlap=False)
    print("\nNon-overlapping 6-mer:")
    print(tokens)
    print("Number of tokens:", len(tokens))
    print("Compression ratio:", len(tokens) / len(sequence))

    tokens = fixed_kmer_tokenize(sequence, k=6, overlap=True)
    print("\nOverlapping 6-mer:")
    print(tokens)
    print("Number of tokens:", len(tokens))
    print("Compression ratio:", len(tokens) / len(sequence))


if __name__ == "__main__":
    main()
