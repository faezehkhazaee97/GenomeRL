"""RL agent for adaptive token gating."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from src.tokenization.boundary_utils import apply_boundary_mask


BASE_TO_ID: Dict[str, int] = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}

ID_TO_BASE: Dict[int, str] = {
    0: "A",
    1: "C",
    2: "G",
    3: "T",
}


def encode_dna_sequence(sequence: str) -> torch.Tensor:
    """Convert a single DNA string into base IDs."""
    sequence = sequence.upper().strip()
    ids = []
    for base in sequence:
        if base not in BASE_TO_ID:
            raise ValueError(f"Invalid base '{base}' in sequence: {sequence}")
        ids.append(BASE_TO_ID[base])
    return torch.tensor(ids, dtype=torch.long)


def encode_dna_batch(sequences: List[str], max_length: int = None):
    """Convert a list of DNA sequences into padded IDs and attention masks."""
    if max_length is None:
        max_length = max(len(sequence) for sequence in sequences)

    batch_ids = []
    batch_mask = []
    for sequence in sequences:
        ids = encode_dna_sequence(sequence).tolist()
        if len(ids) > max_length:
            ids = ids[:max_length]

        mask = [1] * len(ids)
        if len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [0] * pad_len
            mask = mask + [0] * pad_len

        batch_ids.append(ids)
        batch_mask.append(mask)

    input_ids = torch.tensor(batch_ids, dtype=torch.long)
    attention_mask = torch.tensor(batch_mask, dtype=torch.float)
    return input_ids, attention_mask


class TokenGatingAgent(nn.Module):
    """Lightweight sequence model that predicts token-boundary probabilities."""

    def __init__(
        self,
        vocab_size: int = 4,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        initial_boundary_prob: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        self.boundary_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        final_layer = self.boundary_head[-1]
        logit = torch.log(torch.tensor(initial_boundary_prob / (1.0 - initial_boundary_prob)))
        nn.init.constant_(final_layer.bias, float(logit))

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        encoded, _ = self.encoder(x)
        boundary_logits = self.boundary_head(encoded).squeeze(-1)

        if attention_mask is not None:
            boundary_logits = boundary_logits.masked_fill(attention_mask == 0, -1e9)

        boundary_probs = torch.sigmoid(boundary_logits)
        return boundary_logits, boundary_probs


def sample_boundary_mask(boundary_probs: torch.Tensor, attention_mask: torch.Tensor):
    """Sample binary boundary decisions from Bernoulli probabilities."""
    dist = torch.distributions.Bernoulli(probs=boundary_probs)
    masks = dist.sample()
    log_probs = dist.log_prob(masks)

    # Avoid in-place writes so autograd can reuse the original tensors safely.
    first_boundary = torch.zeros_like(masks)
    first_boundary[:, 0] = 1.0
    masks = torch.maximum(masks, first_boundary)
    masks = masks * attention_mask
    log_probs = log_probs * attention_mask
    return masks.long(), log_probs


def masks_to_token_lists(sequences: List[str], masks: torch.Tensor):
    """Convert a batch of masks into variable-length token lists."""
    token_lists = []
    masks_cpu = masks.detach().cpu().tolist()
    for sequence, mask in zip(sequences, masks_cpu):
        valid_mask = mask[: len(sequence)]
        tokens = apply_boundary_mask(sequence, valid_mask)
        token_lists.append(tokens)
    return token_lists


def threshold_boundary_mask(
    boundary_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    threshold: float = 0.5,
):
    """Convert probabilities into deterministic boundary decisions."""
    masks = (boundary_probs >= threshold).long()
    first_boundary = torch.zeros_like(masks)
    first_boundary[:, 0] = 1
    masks = torch.maximum(masks, first_boundary)
    masks = masks * attention_mask.long()
    return masks


def main() -> None:
    sequences = [
        "ATATAAGCGT",
        "ACGTACGTAA",
    ]
    input_ids, attention_mask = encode_dna_batch(sequences)
    agent = TokenGatingAgent()
    logits, probs = agent(input_ids, attention_mask)
    masks, log_probs = sample_boundary_mask(probs, attention_mask)
    token_lists = masks_to_token_lists(sequences, masks)

    print("Input IDs:")
    print(input_ids)
    print()
    print("Attention mask:")
    print(attention_mask)
    print()
    print("Boundary probabilities:")
    print(probs)
    print()
    print("Sampled masks:")
    print(masks)
    print()
    print("Log probabilities:")
    print(log_probs)
    print()
    print("Token lists:")
    for sequence, mask, tokens in zip(sequences, masks.tolist(), token_lists):
        print("Sequence:", sequence)
        print("Mask:    ", mask[: len(sequence)])
        print("Tokens:  ", tokens)
        print()
    print("Shape:", probs.shape)


if __name__ == "__main__":
    main()
