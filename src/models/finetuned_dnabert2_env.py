"""Partially fine-tunable DNABERT-2 environment for GenomeRL."""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BertConfig

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.utils.config import load_config


def masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over valid positions."""
    expanded_mask = mask.unsqueeze(-1).float()
    denom = expanded_mask.sum(dim=1).clamp_min(1.0)
    return (states * expanded_mask).sum(dim=1) / denom


def filter_compatible_state_dict(module: nn.Module, checkpoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep only checkpoint tensors whose keys and shapes match the target module."""
    current_state = module.state_dict()
    compatible = {}
    skipped = []
    for key, value in checkpoint.items():
        if key not in current_state:
            continue
        if current_state[key].shape != value.shape:
            skipped.append((key, tuple(value.shape), tuple(current_state[key].shape)))
            continue
        compatible[key] = value

    if skipped:
        for key, old_shape, new_shape in skipped:
            print(f"Warning: skipping mismatched fine-tuned env key {key}: checkpoint {old_shape} vs model {new_shape}")
    return compatible


class FineTunedDNABERT2Environment(nn.Module):
    """DNABERT-2 backbone with selective unfreezing for grouped RL training."""

    def __init__(
        self,
        backbone_name: str,
        checkpoint_path: str,
        hidden_size: int,
        num_labels: int,
        max_length: int = 512,
        cls_mix_weight: float = 0.0,
        trainable_encoder_layers: int = 4,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        self.max_length = max_length
        self.cls_mix_weight = cls_mix_weight
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.backbone_name = backbone_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone_name,
            trust_remote_code=True,
        )
        config = BertConfig.from_pretrained(backbone_name)
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            config=config,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )

        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is not None:
            for layer in layers:
                attention = getattr(layer, "attention", None)
                self_attention = getattr(attention, "self", None)
                if self_attention is not None and hasattr(self_attention, "p_dropout"):
                    self_attention.p_dropout = 1e-8

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        adapted_checkpoint = {}
        for key, value in checkpoint.items():
            if key.startswith("backbone.") or key.startswith("classifier."):
                adapted_checkpoint[key] = value
        compatible_checkpoint = filter_compatible_state_dict(self, adapted_checkpoint)
        missing, unexpected = self.load_state_dict(compatible_checkpoint, strict=False)
        if missing:
            print(f"Warning: missing fine-tuned env keys: {missing}")
        if unexpected:
            print(f"Warning: unexpected fine-tuned env keys: {unexpected}")

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        if freeze_embeddings:
            embeddings = getattr(self.backbone, "embeddings", None)
            if embeddings is not None:
                for parameter in embeddings.parameters():
                    parameter.requires_grad = False

        if layers is not None:
            trainable_encoder_layers = max(0, min(trainable_encoder_layers, len(layers)))
            if trainable_encoder_layers > 0:
                for layer in layers[-trainable_encoder_layers:]:
                    for parameter in layer.parameters():
                        parameter.requires_grad = True

        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def trainable_parameters(self):
        """Yield parameters that remain trainable."""
        for parameter in self.parameters():
            if parameter.requires_grad:
                yield parameter

    def encode_sequences(self, sequences: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        special_tokens_mask = encoded["special_tokens_mask"].to(device).bool()

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        cls_states = hidden[:, 0, :]
        token_states = hidden[:, 1:, :]
        token_mask = attention_mask[:, 1:].bool() & ~special_tokens_mask[:, 1:]
        token_counts = token_mask.sum(dim=1).float()
        return {
            "cls_states": cls_states,
            "token_states": token_states,
            "token_mask": token_mask.float(),
            "token_counts": token_counts,
        }

    def logits_from_segments(
        self,
        cls_states: torch.Tensor,
        segment_states: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled_segments = masked_mean(segment_states, segment_mask)
        if self.cls_mix_weight <= 0.0:
            sequence_repr = pooled_segments
        else:
            sequence_repr = (
                self.cls_mix_weight * cls_states
                + (1.0 - self.cls_mix_weight) * pooled_segments
            )
        return self.classifier(sequence_repr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_finetuned_env.yaml")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_config = config["env"]
    env = FineTunedDNABERT2Environment(
        backbone_name=env_config["backbone_name"],
        checkpoint_path=env_config["checkpoint_path"],
        hidden_size=env_config["hidden_size"],
        num_labels=env_config["num_labels"],
        max_length=env_config.get("max_length", 512),
        cls_mix_weight=env_config.get("cls_mix_weight", 0.0),
        trainable_encoder_layers=env_config.get("trainable_encoder_layers", 4),
        freeze_embeddings=env_config.get("freeze_embeddings", True),
    ).to(device)

    dataset = GUEPromoterDataset(config["dataset"][f"{args.split}_path"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    features = env.encode_sequences(batch["sequences"], device)

    print("Fine-tuned DNABERT-2 env smoke test")
    print("=" * 80)
    print("Using device:", device)
    print("Token states shape:", tuple(features["token_states"].shape))
    print("Token counts:", features["token_counts"].tolist())
    trainable = sum(parameter.numel() for parameter in env.trainable_parameters())
    print("Trainable parameters:", trainable)


if __name__ == "__main__":
    main()
