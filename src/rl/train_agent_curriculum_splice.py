"""Curriculum RL for splice-site prediction.

Addresses reviewer Q5: combats over-compression on boundary-sensitive tasks via:
  1. Curriculum β schedule: starts at β_start (loose), linearly increases to β_end
  2. Asymmetric class weights: penalize FN (missed splice sites) more than FP
  3. Minimum compression floor raised to 0.15 to prevent early collapse

Curriculum schedule:
  β(t) = β_start + (β_end - β_start) * (t / max_steps)

Asymmetric recall reward:
  class_weights = [1.0 (neg), recall_weight (pos)]
  This makes the policy prefer segmentations that preserve splice-site signal.
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
from src.models.token_state_boundary_agent import TokenStateBoundaryAgent
from src.models.transformer_boundary_agent import TransformerBoundaryAgent
from src.models.token_gating_agent import sample_boundary_mask
from src.rl.grpo import compute_group_advantages, compute_policy_loss
from src.rl.reward import compute_reward_from_loss
from src.rl.rollout import group_token_states
from src.utils.config import load_config
from src.utils.seed import set_seed


def curriculum_beta(step: int, max_steps: int, beta_start: float, beta_end: float) -> float:
    """Linearly anneal β from beta_start to beta_end over training."""
    if max_steps <= 1:
        return beta_end
    progress = min(1.0, step / max_steps)
    return beta_start + (beta_end - beta_start) * progress


def compute_reward(
    classification_loss: torch.Tensor,
    num_segments: torch.Tensor,
    original_token_count: torch.Tensor,
    beta: float,
    min_segment_ratio: float,
    below_min_penalty: float,
) -> Dict[str, torch.Tensor]:
    reward_dict = compute_reward_from_loss(
        classification_loss=classification_loss,
        num_tokens=num_segments,
        sequence_length=original_token_count,
        beta=beta,
    )
    shortfall = (min_segment_ratio - reward_dict["compression_ratio"]).clamp_min(0.0)
    reward = reward_dict["reward"] - below_min_penalty * shortfall
    return {**reward_dict, "reward": reward}


def score_group_samples(
    cls_states, token_states, token_mask, labels,
    boundary_probs, env, group_size, beta,
    min_segment_ratio, below_min_penalty, class_weights,
):
    rewards, losses, log_probs, num_segments = [], [], [], []

    for _ in range(group_size):
        masks, sample_log_probs = sample_boundary_mask(boundary_probs, token_mask)
        grouped = group_token_states(token_states, masks, token_mask)
        logits = env.logits_from_segments(
            cls_states=cls_states,
            segment_states=grouped["segment_states"],
            segment_mask=grouped["segment_mask"],
        )
        w = class_weights.to(logits.device)
        group_losses = F.cross_entropy(logits, labels, weight=w, reduction="none")

        reward_dict = compute_reward(
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
    parser.add_argument("--config", type=str,
                        default="configs/grpo_curriculum_splice.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--use-transformer", action="store_true",
                        help="Use TransformerBoundaryAgent instead of BiLSTM")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["seed"]
    set_seed(seed)

    training_config = config.get("training", {})
    batch_size = args.batch_size or training_config.get("batch_size", 8)
    max_steps = args.max_steps or training_config.get("max_steps", 500)
    log_interval = training_config.get("log_interval", 10)
    output_dir = Path(
        args.output_dir or training_config.get("output_dir",
                                               "results/rl_runs/grpo_curriculum_splice")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Curriculum β parameters
    rl_config = config["rl"]
    beta_start = rl_config.get("beta_start", 0.005)
    beta_end = rl_config.get("beta_end", 0.05)
    min_segment_ratio = rl_config.get("min_segment_ratio", 0.15)
    below_min_penalty = rl_config.get("below_min_penalty", 10.0)
    recall_weight = rl_config.get("recall_weight", 2.0)
    group_size = rl_config.get("group_size", 8)
    num_labels = config["env"].get("num_labels", 2)

    # Build asymmetric class weights: class 0 = no-splice (down-weight),
    # classes 1+ = splice signals (up-weight to encourage recall)
    if num_labels == 3:
        class_weights = torch.tensor([1.0, recall_weight, recall_weight])
        print(f"Class weights (3-class): [{1.0}, {recall_weight}, {recall_weight}]")
    else:
        class_weights = torch.tensor([1.0, recall_weight])
        print(f"Class weights: neg={1.0:.1f}, pos={recall_weight:.1f}")
    print(f"Curriculum β: {beta_start} → {beta_end} over {max_steps} steps")
    print(f"Min segment ratio: {min_segment_ratio}")

    dataset = GUEPromoterDataset(config["dataset"]["train_path"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    env_config = config["env"]
    env = FineTunedDNABERT2Environment(
        backbone_name=env_config["backbone_name"],
        checkpoint_path=env_config["checkpoint_path"],
        hidden_size=env_config["hidden_size"],
        num_labels=env_config["num_labels"],
        max_length=env_config.get("max_length", 512),
        cls_mix_weight=env_config.get("cls_mix_weight", 0.0),
        trainable_encoder_layers=env_config.get("trainable_encoder_layers", 2),
        freeze_embeddings=env_config.get("freeze_embeddings", True),
    ).to(device)

    agent_config = config["agent"]
    use_transformer = args.use_transformer or agent_config.get("use_transformer", False)

    if use_transformer:
        agent = TransformerBoundaryAgent(
            input_dim=agent_config["input_dim"],
            model_dim=agent_config.get("model_dim", 256),
            num_layers=agent_config.get("num_layers", 2),
            num_heads=agent_config.get("num_heads", 4),
            ffn_dim=agent_config.get("ffn_dim", 512),
            dropout=agent_config.get("dropout", 0.1),
            initial_boundary_prob=agent_config.get("initial_boundary_prob", 0.15),
        ).to(device)
        print("Using TransformerBoundaryAgent")
    else:
        agent = TokenStateBoundaryAgent(
            input_dim=agent_config["input_dim"],
            model_dim=agent_config.get("model_dim", 256),
            hidden_dim=agent_config.get("hidden_dim", 128),
            num_layers=agent_config.get("num_layers", 1),
            dropout=agent_config.get("dropout", 0.1),
            initial_boundary_prob=agent_config.get("initial_boundary_prob", 0.15),
        ).to(device)
        print("Using BiLSTM TokenStateBoundaryAgent")

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent parameters: {total_params:,}")

    policy_lr = rl_config.get("policy_learning_rate", 3e-5)
    env_lr = rl_config.get("env_learning_rate", 1e-5)
    optimizer = torch.optim.Adam([
        {"params": list(agent.parameters()), "lr": policy_lr},
        {"params": list(env.trainable_parameters()), "lr": env_lr},
    ])

    history = []
    step = 0

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            labels = batch["labels"].to(device)
            features = env.encode_sequences(batch["sequences"], device)
            cls_states = features["cls_states"]
            token_states = features["token_states"]
            token_mask = features["token_mask"]

            # Compute current β from curriculum schedule
            beta = curriculum_beta(step, max_steps, beta_start, beta_end)

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
                class_weights=class_weights,
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
            torch.nn.utils.clip_grad_norm_(
                list(agent.parameters()) + list(env.trainable_parameters()), 1.0
            )
            optimizer.step()
            step += 1

            if step % log_interval == 0:
                avg_comp = (
                    scored["num_segments"].float().mean()
                    / token_mask.sum(dim=1).float().mean()
                ).item()
                print(
                    f"Step {step}/{max_steps} | β={beta:.4f} | "
                    f"loss={classifier_loss.item():.4f} | "
                    f"policy={policy_loss.item():.4f} | "
                    f"comp={avg_comp:.3f}"
                )
                history.append({
                    "step": step, "beta": beta,
                    "classifier_loss": classifier_loss.item(),
                    "policy_loss": policy_loss.item(),
                    "compression": avg_comp,
                })

    # Save checkpoint
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "env_state_dict": env.state_dict(),
        "config": config,
        "history": history,
    }, output_dir / "agent_env_last.pt")

    # Evaluate on test set
    agent.eval()
    env.eval()
    test_dataset = GUEPromoterDataset(config["dataset"]["test_path"])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    all_preds, all_labels = [], []
    total_orig, total_segs = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            feats = env.encode_sequences(batch["sequences"], device)
            labs = batch["labels"].to(device)
            _, bprobs = agent(feats["token_states"], feats["token_mask"])
            masks, _ = sample_boundary_mask(bprobs, feats["token_mask"])
            grouped = group_token_states(feats["token_states"], masks, feats["token_mask"])
            logits = env.logits_from_segments(
                feats["cls_states"], grouped["segment_states"], grouped["segment_mask"]
            )
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labs.cpu().tolist())
            total_orig += feats["token_mask"].sum().item()
            total_segs += grouped["num_segments"].sum().item()

    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    comp = total_segs / max(1, total_orig)
    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(f1, 4),
        "compression_ratio": round(comp, 4),
    }
    print(f"\nTest: accuracy={accuracy:.4f}  macro_f1={f1:.4f}  compression={comp:.4f}")

    with open(output_dir / "eval_test_sampled.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(output_dir / "train_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
