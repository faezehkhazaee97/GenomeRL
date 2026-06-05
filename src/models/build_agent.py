"""Factory for GenomeRL token-gating agents."""

from __future__ import annotations

from src.models.token_gating_agent import TokenGatingAgent


def build_token_gating_agent(config):
    agent_config = config["agent"]
    return TokenGatingAgent(
        vocab_size=agent_config.get("vocab_size", 4),
        embedding_dim=agent_config.get("embedding_dim", 32),
        hidden_dim=agent_config.get("hidden_dim", 128),
        num_layers=agent_config.get("num_layers", 1),
        dropout=agent_config.get("dropout", 0.1),
        initial_boundary_prob=agent_config.get("initial_boundary_prob", 0.2),
    )
