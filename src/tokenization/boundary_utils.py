"""Boundary utilities for token segmentation."""

from __future__ import annotations

from typing import List


def validate_boundary_mask(sequence: str, mask: List[int]) -> None:
    """Validate that a boundary mask matches a DNA sequence."""
    if len(sequence) != len(mask):
        raise ValueError(
            f"Sequence length {len(sequence)} does not match mask length {len(mask)}."
        )
    if sequence and mask[0] != 1:
        raise ValueError("First boundary value must be 1.")
    for value in mask:
        if value not in (0, 1):
            raise ValueError("Boundary mask must contain only 0 or 1.")


def apply_boundary_mask(sequence: str, mask: List[int]) -> List[str]:
    """Convert a boundary mask into variable-length DNA tokens."""
    sequence = sequence.upper().strip()
    if not sequence:
        return []

    validate_boundary_mask(sequence, mask)
    tokens = []
    current = ""
    for base, boundary in zip(sequence, mask):
        if boundary == 1:
            if current:
                tokens.append(current)
            current = base
        else:
            current += base

    if current:
        tokens.append(current)
    return tokens


def tokens_to_boundary_mask(sequence: str, tokens: List[str]) -> List[int]:
    """Convert tokens back into a boundary mask."""
    sequence = sequence.upper().strip()
    joined = "".join(tokens).upper()
    if joined != sequence:
        raise ValueError("Tokens do not reconstruct the original sequence.")

    mask = []
    for token in tokens:
        for index, _ in enumerate(token):
            mask.append(1 if index == 0 else 0)

    validate_boundary_mask(sequence, mask)
    return mask


def compression_ratio(sequence: str, tokens: List[str]) -> float:
    """Compute token-count compression ratio for a tokenized sequence."""
    sequence = sequence.upper().strip()
    if not sequence:
        return 0.0
    return len(tokens) / len(sequence)


def average_token_length(tokens: List[str]) -> float:
    """Compute average token length."""
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)


def main() -> None:
    sequence = "ATATAAGCGT"
    mask = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    tokens = apply_boundary_mask(sequence, mask)
    recovered_mask = tokens_to_boundary_mask(sequence, tokens)
    print("Sequence:", sequence)
    print("Mask:", mask)
    print("Tokens:", tokens)
    print("Recovered mask:", recovered_mask)
    print("Compression ratio:", compression_ratio(sequence, tokens))
    print("Average token length:", average_token_length(tokens))


if __name__ == "__main__":
    main()
