"""Training entrypoint for the GenomeRL token-gating agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.models.build_agent import build_token_gating_agent
from src.models.build_classifier import build_learned_token_classifier
from src.models.learned_token_classifier import encode_token_lists
from src.models.token_gating_agent import encode_dna_batch
from src.rl.grpo import (
    compute_group_advantages,
    compute_policy_loss,
    sample_group_tokenizations,
    summarize_group_samples,
)
from src.rl.reward import compute_reward_from_loss, summarize_reward_dict
from src.utils.config import load_config
from src.utils.seed import set_seed


def move_tensor_dict_to_device(batch: Tuple[torch.Tensor, ...], device: torch.device):
    return tuple(tensor.to(device) for tensor in batch)


def score_tokenizations(
    token_lists_per_group: List[List[List[str]]],
    labels: torch.Tensor,
    classifier: torch.nn.Module,
    beta: float,
    sequence_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Score grouped tokenizations with the learned-token classifier."""
    batch_size = len(token_lists_per_group)
    group_size = len(token_lists_per_group[0])

    rewards = []
    losses = []

    for group_index in range(group_size):
        token_lists = [token_lists_per_group[batch_index][group_index] for batch_index in range(batch_size)]
        token_ids, token_mask, base_mask = encode_token_lists(token_lists)
        token_ids, token_mask, base_mask = move_tensor_dict_to_device(
            (token_ids, token_mask, base_mask),
            device,
        )
        logits = classifier(token_ids, token_mask, base_mask)
        group_losses = F.cross_entropy(logits, labels, reduction="none")
        num_tokens = torch.tensor(
            [len(tokens) for tokens in token_lists],
            dtype=torch.float,
            device=device,
        )
        reward_dict = compute_reward_from_loss(
            classification_loss=group_losses,
            num_tokens=num_tokens,
            sequence_length=sequence_length,
            beta=beta,
        )
        rewards.append(reward_dict["reward"])
        losses.append(group_losses)

    rewards_tensor = torch.stack(rewards, dim=1)
    losses_tensor = torch.stack(losses, dim=1)
    return {
        "rewards": rewards_tensor,
        "losses": losses_tensor,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
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
    output_dir = Path(args.output_dir or training_config.get("output_dir", "results/rl_runs/grpo"))
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

    agent = build_token_gating_agent(config).to(device)
    classifier = build_learned_token_classifier(config).to(device)

    learning_rate = config["rl"].get("learning_rate", 3e-5)
    optimizer = torch.optim.Adam(
        list(agent.parameters()) + list(classifier.parameters()),
        lr=learning_rate,
    )

    group_size = config["rl"].get("group_size", 8)
    beta = config["rl"].get("beta", 0.05)

    history = []
    step = 0
    torch.autograd.set_detect_anomaly(True, check_nan=False)

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            sequences = batch["sequences"]
            labels = batch["labels"].to(device)
            sequence_length = len(sequences[0])

            input_ids, attention_mask = encode_dna_batch(sequences)
            input_ids, attention_mask = move_tensor_dict_to_device((input_ids, attention_mask), device)

            _, boundary_probs = agent(input_ids, attention_mask)
            grouped_samples = sample_group_tokenizations(
                sequences=sequences,
                boundary_probs=boundary_probs,
                attention_mask=attention_mask,
                group_size=group_size,
            )

            scored = score_tokenizations(
                token_lists_per_group=grouped_samples["token_lists"],
                labels=labels,
                classifier=classifier,
                beta=beta,
                sequence_length=sequence_length,
                device=device,
            )

            rewards = scored["rewards"]
            classification_losses = scored["losses"]
            advantages = compute_group_advantages(rewards)
            policy_loss = compute_policy_loss(
                log_probs=grouped_samples["log_probs"],
                attention_mask=attention_mask,
                advantages=advantages,
            )
            classifier_loss = classification_losses.mean()
            total_loss = classifier_loss + policy_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            reward_summary = summarize_reward_dict(
                compute_reward_from_loss(
                    classification_loss=classification_losses.mean(dim=1),
                    num_tokens=grouped_samples["num_tokens"].mean(dim=1),
                    sequence_length=sequence_length,
                    beta=beta,
                )
            )
            sample_summary = summarize_group_samples(grouped_samples)

            record = {
                "step": step + 1,
                "total_loss": float(total_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "classifier_loss": float(classifier_loss.item()),
                **reward_summary,
                **sample_summary,
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
    checkpoint_path = output_dir / "agent_classifier_last.pt"
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
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
