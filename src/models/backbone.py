"""Model backbone definitions."""

from __future__ import annotations

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertConfig


class FrozenBackboneClassifier(nn.Module):
    """Frozen transformer backbone with a small trainable classifier head."""

    def __init__(
        self,
        backbone_name: str,
        hidden_size: int,
        num_labels: int,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained(
            backbone_name,
            trust_remote_code=True,
        )
        config = BertConfig.from_pretrained(backbone_name)
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            config=config,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )

        # DNABERT-2 remote code uses a Triton FlashAttention path only when
        # attention dropout is exactly zero. A tiny nonzero value forces the
        # built-in PyTorch attention fallback, which is more robust on this
        # cluster toolchain.
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is not None:
            for layer in layers:
                attention = getattr(layer, "attention", None)
                self_attention = getattr(attention, "self", None)
                if self_attention is not None and hasattr(self_attention, "p_dropout"):
                    self_attention.p_dropout = 1e-8

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        pooled = hidden[:, 0, :]
        logits = self.classifier(pooled)
        return logits
