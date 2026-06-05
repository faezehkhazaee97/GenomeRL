"""Lagrangian-constrained RL training for adaptive DNA tokenization.

Replaces the soft β penalty with a hard compression budget enforced via
dual ascent (Lagrangian relaxation). The dual variable λ is updated each
step to push the policy toward the target compression ratio C_target.

Primal update:  maximize  -L_cls - λ * C(b)        (policy gradient)
Dual update:    λ ← max(0, λ + α_λ * (C(b) - C_target))

At convergence, C(b) ≈ C_target and λ is the shadow price of the budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.models.finetuned_dnabert2_env import FineTunedDNABERT2Environment
from src.models.token_gating_agent import sample_boundary_mask
from src.models.token_state_boundary_agent import TokenStateBoundaryAgent
from src.rl.grpo import compute_group_advantages, compute_policy_loss
from src.rl.reward import compute_reward_from_loss, summarize_reward_dict
from src.rl.rollout import group_token_states, summarize_segment_samples
from src.utils.config import load_config
from src.utils.seed import set_seed


def score_group_samples(
    cls_states: torch.Tensor,
    token_states: torch.Tensor,
    token_mask: torch.Tensor,
    labels: torch.Tensor,
    boundary_probs: torch.Tensor,
    env: FineTunedDNABERT2Environment,
    group_size: int,
    lambda_dual: float,
    target_compression: float,
    min_segment_ratio: float,
    below_min_penalty: float,
) -> Dict[str, torch.Tensor]:
    rewards, losses, log_probs, num_segments = [], [], [], []

    for _ in range(group_size):
        masks, sample_log_probs = sample_boundary_mask(boundary_probs, token_mask)
        grouped = group_token_states(token_states, masks, token_mask)
        logits = env.logits_from_segments(
            cls_states=cls_states,
            segment_states=grouped["segment_states"],
            segment_mask=grouped["segment_mask"],
        )
        group_losses = F.cross_entropy(logits, labels, reduction="none")

        comp_ratio = grouped["num_segments"] / token_mask.sum(dim=1).clamp_min(1.0)

        # Lagrangian reward: -L_cls - λ * C(b)
        reward = -group_losses.detach() - lambda_dual * comp_ratio.detach()

        # Soft floor penalty to prevent complete collapse
        shortfall = (min_segment_ratio - comp_ratio).clamp_min(0.0)
        reward = reward - below_min_penalty * shortfall.detach()

        rewards.append(reward)
        losses.append(group_losses)
        log_probs.append(sample_log_probs)
        num_segments.append(grouped["num_segments"])

    return {
        "rewards": torch.stack(rewards, dim=1),  # (batch, group_size)
        "losses": torch.stack(losses, dim=1),  # (batch, group_size)
        "log_probs": torch.stack(log_probs, dim=1),  # (batch, group_size, seq_len)
        "token_mask": token_mask,  # (batch, seq_len)
        "num_segments": torch.stack(num_segments, dim=1),  # (batch, group_size)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = config["env"]
    agent_cfg = config["agent"]
    rl_cfg = config["rl"]
    lag_cfg = config.get("lagrangian", {})

    target_compression = lag_cfg.get("target_compression", 0.18)
    dual_lr = lag_cfg.get("dual_lr", 0.01)
    lambda_dual = lag_cfg.get("lambda_init", 0.5)

    env = FineTunedDNABERT2Environment(
        backbone_name=env_cfg["backbone_name"],
        checkpoint_path=env_cfg["checkpoint_path"],
        hidden_size=env_cfg["hidden_size"],
        num_labels=env_cfg["num_labels"],
        max_length=env_cfg.get("max_length", 512),
        cls_mix_weight=env_cfg.get("cls_mix_weight", 0.0),
        trainable_encoder_layers=env_cfg.get("trainable_encoder_layers", 2),
        freeze_embeddings=env_cfg.get("freeze_embeddings", True),
    ).to(device)

    agent = TokenStateBoundaryAgent(
        input_dim=agent_cfg["input_dim"],
        model_dim=agent_cfg.get("model_dim", 256),
        hidden_dim=agent_cfg.get("hidden_dim", 128),
        num_layers=agent_cfg.get("num_layers", 1),
        dropout=agent_cfg.get("dropout", 0.1),
        initial_boundary_prob=agent_cfg.get("initial_boundary_prob", 0.2),
    ).to(device)

    policy_optimizer = torch.optim.AdamW(
        agent.parameters(), lr=float(rl_cfg["policy_learning_rate"])
    )
    env_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, env.parameters()),
        lr=float(rl_cfg["env_learning_rate"]),
    )

    dataset = GUEPromoterDataset(config["dataset"]["train_path"])
    loader = DataLoader(
        dataset,
        batch_size=config["training"].get("batch_size", 8),
        shuffle=True,
        collate_fn=collate_fn,
    )

    max_steps = config["training"]["max_steps"]
    log_interval = config["training"].get("log_interval", 10)
    group_size = rl_cfg.get("group_size", 8)
    min_segment_ratio = rl_cfg.get("min_segment_ratio", 0.08)
    below_min_penalty = rl_cfg.get("below_min_penalty", 5.0)

    history = []
    data_iter = iter(loader)
    step = 0

    print(f"Target compression: {target_compression:.3f}  lambda_init: {lambda_dual:.3f}")

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        sequences = batch["sequences"]
        labels = batch["labels"].to(device)

        features = env.encode_sequences(sequences, device)
        cls_states = features["cls_states"]
        token_states = features["token_states"]
        token_mask = features["token_mask"]

        _, boundary_probs = agent(token_states, token_mask)

        samples = score_group_samples(
            cls_states=cls_states,
            token_states=token_states,
            token_mask=token_mask,
            labels=labels,
            boundary_probs=boundary_probs,
            env=env,
            group_size=group_size,
            lambda_dual=lambda_dual,
            target_compression=target_compression,
            min_segment_ratio=min_segment_ratio,
            below_min_penalty=below_min_penalty,
        )

        advantages = compute_group_advantages(samples["rewards"])
        policy_loss = compute_policy_loss(
            samples["log_probs"], samples["token_mask"], advantages
        )
        env_loss = samples["losses"].mean()

        policy_optimizer.zero_grad()
        env_optimizer.zero_grad()

        # Backward on both losses together to avoid graph clearing
        total_loss = policy_loss + env_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(env.parameters(), 1.0)

        policy_optimizer.step()
        env_optimizer.step()

        # Dual ascent: update λ based on constraint violation
        avg_comp = samples["num_segments"].float().mean().item() / max(
            token_mask.sum(dim=1).float().mean().item(), 1e-6
        )
        lambda_dual = max(0.0, lambda_dual + dual_lr * (avg_comp - target_compression))

        step += 1

        if step % log_interval == 0:
            record = {
                "step": step,
                "policy_loss": policy_loss.item(),
                "env_loss": env_loss.item(),
                "avg_compression": avg_comp,
                "lambda_dual": lambda_dual,
                "target_compression": target_compression,
                "constraint_violation": avg_comp - target_compression,
            }
            print(f"Step {step}/{max_steps} | "
                  f"loss={env_loss.item():.4f} | "
                  f"comp={avg_comp:.3f} (target={target_compression:.3f}) | "
                  f"λ={lambda_dual:.3f}")
            history.append(record)

    torch.save({"agent_state_dict": agent.state_dict(),
                "env_state_dict": env.state_dict(),
                "lambda_dual": lambda_dual},
               output_dir / "agent_env_last.pt")

    with open(output_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Done. Final λ={lambda_dual:.3f}, avg_comp={avg_comp:.3f}")


if __name__ == "__main__":
    main()
