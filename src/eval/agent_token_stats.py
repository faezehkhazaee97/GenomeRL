"""Token statistics for learned boundary-based tokenization."""

from __future__ import annotations

from typing import Dict, List

from src.tokenization.boundary_utils import average_token_length, compression_ratio


def summarize_agent_tokens(sequences: List[str], token_lists: List[List[str]]) -> Dict[str, float]:
    token_counts = []
    compression_ratios = []
    avg_token_lengths = []

    for sequence, tokens in zip(sequences, token_lists):
        token_counts.append(len(tokens))
        compression_ratios.append(compression_ratio(sequence, tokens))
        avg_token_lengths.append(average_token_length(tokens))

    return {
        "avg_num_tokens": sum(token_counts) / len(token_counts),
        "avg_compression_ratio": sum(compression_ratios) / len(compression_ratios),
        "avg_token_length": sum(avg_token_lengths) / len(avg_token_lengths),
    }
