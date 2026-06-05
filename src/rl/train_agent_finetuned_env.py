"""Train RL grouping with a partially fine-tuned DNABERT-2 backbone."""

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
from src.models.attention_pooling_env import AttentionPoolingEnv
from src.models.segment_aware_env import SegmentAwareDNABERT2Environment
from src.models.token_gating_agent import sample_boundary_mask
from src.models.token_state_boundary_agent import TokenStateBoundaryAgent
from src.rl.grpo import compute_group_advantages, compute_policy_loss
from src.rl.reward import compute_reward_from_loss, summarize_reward_dict
from src.rl.rollout import group_token_states, summarize_segment_samples
from src.utils.config import load_config
from src.utils.seed import set_seed


def compute_finetuned_reward(
    classification_loss: torch.Tensor,
    num_segments: torch.Tensor,
    original_token_count: torch.Tensor,
    beta: float,
    min_segment_ratio: float,
    below_min_penalty: float,
) -> Dict[str, torch.Tensor]:
    """Reward with an explicit anti-collapse penalty below a minimum ratio."""
    reward_dict = compute_reward_from_loss(
        classification_loss=classification_loss,
        num_tokens=num_segments,
        sequence_length=original_token_count,
        beta=beta,
    )
    shortfall = (min_segment_ratio - reward_dict["compression_ratio"]).clamp_min(0.0)
    min_ratio_penalty = below_min_penalty * shortfall
    reward = reward_dict["reward"] - min_ratio_penalty
    return {
        **reward_dict,
        "min_ratio_penalty": min_ratio_penalty,
        "reward": reward,
    }


def score_group_samples(
    cls_states: torch.Tensor,
    token_states: torch.Tensor,
    token_mask: torch.Tensor,
    labels: torch.Tensor,
    boundary_probs: torch.Tensor,
    env: FineTunedDNABERT2Environment,
    group_size: int,
    beta: float,
    min_segment_ratio: float,
    below_min_penalty: float,
) -> Dict[str, torch.Tensor]:
    rewards = []
    losses = []
    log_probs = []
    num_segments = []

    for _ in range(group_size):
        masks, sample_log_probs = sample_boundary_mask(boundary_probs, token_mask)
        grouped = group_token_states(token_states, masks, token_mask)
        logits = env.logits_from_segments(
            cls_states=cls_states,
            segment_states=grouped["segment_states"],
            segment_mask=grouped["segment_mask"],
        )
        group_losses = F.cross_entropy(logits, labels, reduction="none")
        reward_dict = compute_finetuned_reward(
            classification_loss=group_losses.detach(),
            num_segments=grouped["num_segments"],
            original_token_count=token_mask.sum(dim=1),
            beta=beta,
            min_segment_ratio=min_segment_ratio,
            below_min_penalty=below_min_penalty,
        )

        rewards.append(reward_dict["reward"])
        losses.append(group_losses)
        log_probs.append(sample_log_probs)
        num_segments.append(grouped["num_segments"])

    return {
        "rewards": torch.stack(rewards, dim=1),
        "losses": torch.stack(losses, dim=1),
        "log_probs": torch.stack(log_probs, dim=1),
        "num_segments": torch.stack(num_segments, dim=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_finetuned_env.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["seed"]
    config["seed"] = seed
    set_seed(seed)

    training_config = config.get("training", {})
    batch_size = args.batch_size or training_config.get("batch_size", 8)
    max_steps = args.max_steps or training_config.get("max_steps", 50)
    num_workers = training_config.get("num_workers", 0)
    log_interval = training_config.get("log_interval", 10)
    output_dir = Path(args.output_dir or training_config.get("output_dir", "results/rl_runs/grpo_finetuned_env"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = GUEPromoterDataset(config["dataset"]["train_path"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    env_config = config["env"]

    # Choose environment class based on config flags
    if env_config.get("use_attention_pooling", False):
        EnvClass = AttentionPoolingEnv
    elif env_config.get("use_segment_pe", False):
        EnvClass = SegmentAwareDNABERT2Environment
    else:
        EnvClass = FineTunedDNABERT2Environment

    env_kwargs = {
        "backbone_name": env_config["backbone_name"],
        "checkpoint_path": env_config["checkpoint_path"],
        "hidden_size": env_config["hidden_size"],
        "num_labels": env_config["num_labels"],
        "max_length": env_config.get("max_length", 512),
        "cls_mix_weight": env_config.get("cls_mix_weight", 0.0),
        "trainable_encoder_layers": env_config.get("trainable_encoder_layers", 4),
        "freeze_embeddings": env_config.get("freeze_embeddings", True),
    }

    # Add segment-PE specific kwargs if using SegmentAwareDNABERT2Environment
    if EnvClass == SegmentAwareDNABERT2Environment:
        env_kwargs.update({
            "segment_pe_dim": env_config.get("segment_pe_dim", 128),
            "max_segments": env_config.get("max_segments", 100),
        })

    env = EnvClass(**env_kwargs).to(device)

    agent_config = config["agent"]
    agent = TokenStateBoundaryAgent(
        input_dim=agent_config["input_dim"],
        model_dim=agent_config.get("model_dim", 256),
        hidden_dim=agent_config.get("hidden_dim", 128),
        num_layers=agent_config.get("num_layers", 1),
        dropout=agent_config.get("dropout", 0.1),
        initial_boundary_prob=agent_config.get("initial_boundary_prob", 0.2),
    ).to(device)

    policy_lr = config["rl"].get("policy_learning_rate", 3e-5)
    env_lr = config["rl"].get("env_learning_rate", 1e-5)
    optimizer = torch.optim.Adam(
        [
            {"params": list(agent.parameters()), "lr": policy_lr},
            {"params": list(env.trainable_parameters()), "lr": env_lr},
        ]
    )
    group_size = config["rl"].get("group_size", 8)
    beta = args.beta if args.beta is not None else config["rl"].get("beta", 0.02)
    min_segment_ratio = config["rl"].get("min_segment_ratio", 0.08)
    below_min_penalty = config["rl"].get("below_min_penalty", 5.0)

    history = []
    step = 0

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            labels = batch["labels"].to(device)
            sequences = batch["sequences"]

            features = env.encode_sequences(sequences, device)
            cls_states = features["cls_states"]
            token_states = features["token_states"]
            token_mask = features["token_mask"]

            _, boundary_probs = agent(token_states.detach(), token_mask)
            scored = score_group_samples(
                cls_states=cls_states,
                token_states=token_states,
                token_mask=token_mask,
                labels=labels,
                boundary_probs=boundary_probs,
                env=env,
                group_size=group_size,
                beta=beta,
                min_segment_ratio=min_segment_ratio,
                below_min_penalty=below_min_penalty,
            )

            advantages = compute_group_advantages(scored["rewards"])
            policy_loss = compute_policy_loss(
                log_probs=scored["log_probs"],
                attention_mask=token_mask,
                advantages=advantages,
            )
            classifier_loss = scored["losses"].mean()
            total_loss = classifier_loss + policy_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            reward_summary = summarize_reward_dict(
                compute_finetuned_reward(
                    classification_loss=scored["losses"].mean(dim=1).detach(),
                    num_segments=scored["num_segments"].mean(dim=1),
                    original_token_count=token_mask.sum(dim=1),
                    beta=beta,
                    min_segment_ratio=min_segment_ratio,
                    below_min_penalty=below_min_penalty,
                )
            )
            segment_summary = summarize_segment_samples(scored["num_segments"])
            record = {
                "step": step + 1,
                "total_loss": float(total_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "classifier_loss": float(classifier_loss.item()),
                **reward_summary,
                **segment_summary,
            }
            history.append(record)

            if (step + 1) % log_interval == 0 or step == 0:
                print("=" * 80)
                print(f"Step {step + 1}/{max_steps}")
                print(json.dumps(record, indent=2))

            step += 1

    history_path = output_dir / "train_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    checkpoint_path = output_dir / "agent_env_last.pt"
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "env_state_dict": env.state_dict(),
            "config": config,
            "history": history,
        },
        checkpoint_path,
    )
    print()
    print(f"Saved training history to {history_path}")
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
