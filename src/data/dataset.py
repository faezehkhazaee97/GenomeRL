"""Dataset definitions for GenomeRL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import Dataset


class GUEPromoterDataset(Dataset):
    """Thin JSONL-backed dataset for processed GUE promoter splits."""

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.items = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                self.items.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        return {
            "id": item["id"],
            "sequence": item["sequence"],
            "label": torch.tensor(item["label"], dtype=torch.long),
            "split": item["split"],
        }


def collate_fn(batch):
    return {
        "ids": [item["id"] for item in batch],
        "sequences": [item["sequence"] for item in batch],
        "labels": torch.stack([item["label"] for item in batch]),
        "splits": [item["split"] for item in batch],
    }
