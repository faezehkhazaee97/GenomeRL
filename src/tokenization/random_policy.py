"""Random tokenization policy baselines."""

from __future__ import annotations

import random
from typing import List, Optional

from src.tokenization.boundary_utils import apply_boundary_mask


def random_boundary_mask(
    sequence: str,
    boundary_prob: float = 0.2,
    seed: Optional[int] = None,
) -> List[int]:
    """Generate a random boundary mask for a DNA sequence."""
    if not 0.0 <= boundary_prob <= 1.0:
        raise ValueError("boundary_prob must be between 0 and 1.")
    if seed is not None:
        random.seed(seed)

    sequence = sequence.upper().strip()
    if not sequence:
        return []

    mask = [1]
    for _ in sequence[1:]:
        mask.append(1 if random.random() < boundary_prob else 0)
    return mask


def random_tokenize(
    sequence: str,
    boundary_prob: float = 0.2,
    seed: Optional[int] = None,
) -> List[str]:
    """Tokenize DNA with random boundaries."""
    mask = random_boundary_mask(
        sequence=sequence,
        boundary_prob=boundary_prob,
        seed=seed,
    )
    return apply_boundary_mask(sequence, mask)


def main() -> None:
    sequence = "ATATAAGCGT"
    mask = random_boundary_mask(sequence, boundary_prob=0.3, seed=42)
    tokens = apply_boundary_mask(sequence, mask)
    print("Sequence:", sequence)
    print("Mask:    ", mask)
    print("Tokens:  ", tokens)
    print("Number of tokens:", len(tokens))
    print("Compression ratio:", len(tokens) / len(sequence))


if __name__ == "__main__":
    main()
