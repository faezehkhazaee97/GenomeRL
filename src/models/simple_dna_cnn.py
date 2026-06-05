"""Simple fallback DNA CNN baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


BASE_TO_ID = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


def encode_dna_batch(sequences, max_length=None):
    if max_length is None:
        max_length = max(len(sequence) for sequence in sequences)

    batch_ids = []
    for sequence in sequences:
        sequence = sequence.upper().strip()
        ids = [BASE_TO_ID.get(base, 0) for base in sequence]
        if len(ids) > max_length:
            ids = ids[:max_length]
        if len(ids) < max_length:
            ids = ids + [0] * (max_length - len(ids))
        batch_ids.append(ids)
    return torch.tensor(batch_ids, dtype=torch.long)


class SimpleDNACNN(nn.Module):
    def __init__(
        self,
        num_labels: int = 2,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(4, embedding_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.mean(dim=-1)
        logits = self.classifier(x)
        return logits
