"""Classifier for variable-length DNA token sequences.

This module is designed for the learned-token RL stage:
1. the boundary agent proposes a tokenization
2. this classifier scores the resulting token sequence for promoter prediction
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from src.models.token_gating_agent import BASE_TO_ID


def encode_token_lists(
    token_lists: List[List[str]],
    max_num_tokens: int | None = None,
    max_token_length: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert tokenized DNA sequences into padded tensors.

    Returns:
        token_ids: [batch, max_num_tokens, max_token_length]
        token_mask: [batch, max_num_tokens]
        base_mask: [batch, max_num_tokens, max_token_length]
    """
    if not token_lists:
        raise ValueError("token_lists must not be empty")

    if max_num_tokens is None:
        max_num_tokens = max(len(tokens) for tokens in token_lists)
    if max_token_length is None:
        max_token_length = max(len(token) for tokens in token_lists for token in tokens)

    batch_token_ids = []
    batch_token_mask = []
    batch_base_mask = []

    for tokens in token_lists:
        padded_tokens = []
        padded_base_mask = []

        for token in tokens[:max_num_tokens]:
            token = token.upper().strip()
            token_ids = [BASE_TO_ID.get(base, 0) for base in token[:max_token_length]]
            base_mask = [1] * len(token_ids)

            if len(token_ids) < max_token_length:
                pad_len = max_token_length - len(token_ids)
                token_ids += [0] * pad_len
                base_mask += [0] * pad_len

            padded_tokens.append(token_ids)
            padded_base_mask.append(base_mask)

        token_mask = [1] * len(padded_tokens)

        if len(padded_tokens) < max_num_tokens:
            pad_tokens = max_num_tokens - len(padded_tokens)
            padded_tokens += [[0] * max_token_length for _ in range(pad_tokens)]
            padded_base_mask += [[0] * max_token_length for _ in range(pad_tokens)]
            token_mask += [0] * pad_tokens

        batch_token_ids.append(padded_tokens)
        batch_token_mask.append(token_mask)
        batch_base_mask.append(padded_base_mask)

    token_ids_tensor = torch.tensor(batch_token_ids, dtype=torch.long)
    token_mask_tensor = torch.tensor(batch_token_mask, dtype=torch.float)
    base_mask_tensor = torch.tensor(batch_base_mask, dtype=torch.float)
    return token_ids_tensor, token_mask_tensor, base_mask_tensor


class LearnedTokenClassifier(nn.Module):
    """Promoter classifier that consumes learned DNA tokens."""

    def __init__(
        self,
        num_labels: int = 2,
        vocab_size: int = 4,
        base_embedding_dim: int = 32,
        token_hidden_dim: int = 128,
        sequence_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_embedding = nn.Embedding(vocab_size, base_embedding_dim)
        self.token_encoder = nn.Sequential(
            nn.Conv1d(base_embedding_dim, token_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(token_hidden_dim, token_hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.token_projector = nn.Sequential(
            nn.Linear(token_hidden_dim * 2 + 1, token_hidden_dim),
            nn.LayerNorm(token_hidden_dim),
            nn.Dropout(dropout),
        )
        self.sequence_encoder = nn.LSTM(
            input_size=token_hidden_dim,
            hidden_size=sequence_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention_pool = nn.Sequential(
            nn.Linear(sequence_hidden_dim * 2, sequence_hidden_dim),
            nn.Tanh(),
            nn.Linear(sequence_hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(sequence_hidden_dim * 2, sequence_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sequence_hidden_dim, num_labels),
        )

    def _encode_tokens(
        self,
        token_ids: torch.Tensor,
        base_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode each token with a small CNN over its constituent bases."""
        batch_size, max_num_tokens, max_token_length = token_ids.shape
        flat_token_ids = token_ids.view(batch_size * max_num_tokens, max_token_length)
        flat_base_mask = base_mask.view(batch_size * max_num_tokens, max_token_length)

        base_embeddings = self.base_embedding(flat_token_ids)  # [BN, L, D]
        token_features = self.token_encoder(base_embeddings.transpose(1, 2))  # [BN, H, L]
        token_features = token_features.transpose(1, 2)  # [BN, L, H]

        expanded_mask = flat_base_mask.unsqueeze(-1)
        masked_features = token_features * expanded_mask
        token_lengths = flat_base_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

        mean_pool = masked_features.sum(dim=1) / token_lengths
        max_fill = torch.full_like(token_features, -1e9)
        max_pool = torch.where(expanded_mask.bool(), token_features, max_fill).max(dim=1).values
        max_pool = torch.where(
            token_lengths > 0,
            max_pool,
            torch.zeros_like(max_pool),
        )
        normalized_length = token_lengths / max_token_length

        token_vectors = torch.cat([mean_pool, max_pool, normalized_length], dim=-1)
        token_vectors = self.token_projector(token_vectors)
        return token_vectors.view(batch_size, max_num_tokens, -1)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        base_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_vectors = self._encode_tokens(token_ids, base_mask)
        encoded, _ = self.sequence_encoder(token_vectors)
        attention_scores = self.attention_pool(encoded).squeeze(-1)
        attention_scores = attention_scores.masked_fill(token_mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        pooled = torch.sum(encoded * attention_weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


def main() -> None:
    token_lists = [
        ["ATA", "T", "AAGCG", "T"],
        ["AC", "GTACG", "TA", "A"],
    ]
    token_ids, token_mask, base_mask = encode_token_lists(token_lists)
    model = LearnedTokenClassifier()
    logits = model(token_ids, token_mask, base_mask)

    print("Token IDs shape:", tuple(token_ids.shape))
    print("Token mask shape:", tuple(token_mask.shape))
    print("Base mask shape:", tuple(base_mask.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Logits:")
    print(logits)


if __name__ == "__main__":
    main()
