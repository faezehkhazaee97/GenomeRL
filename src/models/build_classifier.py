"""Factory helpers for learned-token classifiers."""

from __future__ import annotations

from src.models.learned_token_classifier import LearnedTokenClassifier


def build_learned_token_classifier(config):
    classifier_config = config.get("classifier", {})
    return LearnedTokenClassifier(
        num_labels=classifier_config.get("num_labels", 2),
        vocab_size=classifier_config.get("vocab_size", 4),
        base_embedding_dim=classifier_config.get("base_embedding_dim", 32),
        token_hidden_dim=classifier_config.get("token_hidden_dim", 128),
        sequence_hidden_dim=classifier_config.get("sequence_hidden_dim", 128),
        dropout=classifier_config.get("dropout", 0.1),
    )
