"""Splice-site RL with GT/AG proximity reward (reviewer Q5).

Adds a reward bonus for placing boundaries near canonical GT (donor) and
AG (acceptor) splice dinucleotides. Combined with curriculum β annealing
and asymmetric recall weights from the previous experiment.

Reward = -L_cls - β(t)·C(b) + γ·motif_proximity(b, seq) - penalty_if_below_min
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import GUEPromoterDataset, collate_fn
from src.models.finetuned_dnabert2_env import FineTunedDNABERT2Environment
from src.models.token_state_boundary_agent import TokenStateBoundaryAgent
from src.models.token_gating_agent import sample_boundary_mask
from src.rl.grpo import compute_group_advantages, compute_policy_loss
from src.rl.reward import compute_reward_from_loss
from src.rl.rollout import group_token_states
from src.utils.config import load_config
from src.utils.seed import set_seed


def motif_proximity_bonus(
    sequences: List[str],
    boundary_masks: torch.Tensor,
    token_mask: torch.Tensor,
    window: int = 3,
    gamma: float = 0.1,
) -> torch.Tensor:
    """Reward boundaries placed near GT (donor) or AG (acceptor) dinucleotides."""
    B = boundary_masks.size(0)
    bonuses = []
    for b in range(B):
        seq = sequences[b].upper()
        seq_len = int(token_mask[b].sum().item())
        mask_b = boundary_masks[b, :seq_len]
        boundary_pos = mask_b.nonzero(as_tuple=False).squeeze(-1).tolist()
        if not boundary_pos:
            bonuses.append(0.0)
            continue
        nt_per_tok = max(1.0, len(seq) / seq_len)
        near = 0
        for pos in boundary_pos:
            nt = int(pos * nt_per_tok)
            win_nt = int(window * nt_per_tok) + 4
            region = seq[max(0, nt - win_nt): min(len(seq), nt + win_nt + 2)]
            if "GT" in region or "AG" in region:
                near += 1
        bonuses.append(gamma * near / len(boundary_pos))
    return torch.tensor(bonuses, dtype=torch.float32, device=boundary_masks.device)


def curriculum_beta(step, max_steps, beta_start, beta_end):
    return beta_start + (beta_end - beta_start) * min(1.0, step / max(max_steps, 1))


def score_group_samples(
    sequences, cls_states, token_states, token_mask, labels,
    boundary_probs, env, group_size, beta,
    min_segment_ratio, below_min_penalty,
    class_weights, gamma, window,
):
    rewards, losses, log_probs, num_segments = [], [], [], []
    for _ in range(group_size):
        masks, slp = sample_boundary_mask(boundary_probs, token_mask)
        grouped = group_token_states(token_states, masks, token_mask)
        logits = env.logits_from_segments(
            cls_states, grouped["segment_states"], grouped["segment_mask"]
        )
        w = class_weights.to(logits.device)
        group_losses = F.cross_entropy(logits, labels, weight=w, reduction="none")
        rd = compute_reward_from_loss(
            classification_loss=group_losses.detach(),
            num_tokens=grouped["num_segments"],
            sequence_length=token_mask.sum(dim=1),
            beta=beta,
        )
        shortfall = (min_segment_ratio - rd["compression_ratio"]).clamp_min(0.0)
        base_reward = rd["reward"] - below_min_penalty * shortfall
        bonus = motif_proximity_bonus(sequences, masks, token_mask, window, gamma)
        rewards.append(base_reward + bonus)
        losses.append(group_losses)
        log_probs.append(slp)
        num_segments.append(grouped["num_segments"])
    return {
        "rewards": torch.stack(rewards, dim=1),
        "losses": torch.stack(losses, dim=1),
        "log_probs": torch.stack(log_probs, dim=1),
        "num_segments": torch.stack(num_segments, dim=1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_curriculum_splice.yaml")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    tc = config.get("training", {})
    batch_size = tc.get("batch_size", 8)
    max_steps = tc.get("max_steps", 500)
    log_interval = tc.get("log_interval", 10)
    output_dir = Path(args.output_dir or
                      f"results/rl_runs/grpo_splice_motif_gamma{args.gamma}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  gamma={args.gamma}  window={args.window}")

    rc = config["rl"]
    beta_start = rc.get("beta_start", 0.005)
    beta_end = rc.get("beta_end", 0.05)
    min_seg = rc.get("min_segment_ratio", 0.15)
    below_pen = rc.get("below_min_penalty", 10.0)
    recall_w = rc.get("recall_weight", 2.0)
    group_size = rc.get("group_size", 8)
    num_labels = config["env"].get("num_labels", 3)
    class_weights = torch.tensor([1.0, recall_w, recall_w])

    dataset = GUEPromoterDataset(config["dataset"]["train_path"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    ec = config["env"]
    env = FineTunedDNABERT2Environment(
        backbone_name=ec["backbone_name"],
        checkpoint_path=ec["checkpoint_path"],
        hidden_size=ec["hidden_size"],
        num_labels=num_labels,
        max_length=ec.get("max_length", 512),
        cls_mix_weight=0.0,
        trainable_encoder_layers=2,
        freeze_embeddings=True,
    ).to(device)

    ac = config["agent"]
    agent = TokenStateBoundaryAgent(
        input_dim=ac["input_dim"], model_dim=256, hidden_dim=128,
        initial_boundary_prob=0.15,
    ).to(device)

    optimizer = torch.optim.Adam([
        {"params": list(agent.parameters()), "lr": rc.get("policy_learning_rate", 3e-5)},
        {"params": list(env.trainable_parameters()), "lr": rc.get("env_learning_rate", 1e-5)},
    ])

    history = []
    step = 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            labels = batch["labels"].to(device)
            seqs = batch["sequences"]
            feats = env.encode_sequences(seqs, device)
            beta = curriculum_beta(step, max_steps, beta_start, beta_end)
            _, bprobs = agent(feats["token_states"].detach(), feats["token_mask"])
            scored = score_group_samples(
                sequences=seqs,
                cls_states=feats["cls_states"],
                token_states=feats["token_states"],
                token_mask=feats["token_mask"],
                labels=labels,
                boundary_probs=bprobs,
                env=env,
                group_size=group_size,
                beta=beta,
                min_segment_ratio=min_seg,
                below_min_penalty=below_pen,
                class_weights=class_weights,
                gamma=args.gamma,
                window=args.window,
            )
            advantages = compute_group_advantages(scored["rewards"])
            policy_loss = compute_policy_loss(scored["log_probs"], feats["token_mask"], advantages)
            classifier_loss = scored["losses"].mean()
            optimizer.zero_grad()
            (classifier_loss + policy_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.parameters()) + list(env.trainable_parameters()), 1.0)
            optimizer.step()
            step += 1
            if step % log_interval == 0:
                comp = (scored["num_segments"].float().mean() /
                        feats["token_mask"].sum(dim=1).float().mean()).item()
                print(f"Step {step}/{max_steps} | β={beta:.4f} | "
                      f"loss={classifier_loss.item():.4f} | comp={comp:.3f}")
                history.append({"step": step, "beta": beta,
                                 "loss": classifier_loss.item(), "compression": comp})

    torch.save({"agent_state_dict": agent.state_dict(),
                "env_state_dict": env.state_dict(),
                "config": config, "history": history},
               output_dir / "agent_env_last.pt")

    # Evaluate
    agent.eval(); env.eval()
    test_loader = DataLoader(GUEPromoterDataset(config["dataset"]["test_path"]),
                             batch_size=32, shuffle=False, collate_fn=collate_fn)
    preds, labs_all, tot_orig, tot_segs = [], [], 0, 0
    with torch.no_grad():
        for batch in test_loader:
            feats = env.encode_sequences(batch["sequences"], device)
            labs = batch["labels"].to(device)
            _, bp = agent(feats["token_states"], feats["token_mask"])
            masks, _ = sample_boundary_mask(bp, feats["token_mask"])
            grouped = group_token_states(feats["token_states"], masks, feats["token_mask"])
            logits = env.logits_from_segments(
                feats["cls_states"], grouped["segment_states"], grouped["segment_mask"])
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs_all.extend(labs.cpu().tolist())
            tot_orig += feats["token_mask"].sum().item()
            tot_segs += grouped["num_segments"].sum().item()

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(labs_all, preds)
    f1 = f1_score(labs_all, preds, average="macro", zero_division=0)
    comp = tot_segs / max(1, tot_orig)
    metrics = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4),
               "compression_ratio": round(comp, 4), "gamma": args.gamma}
    print(f"\nTest: acc={acc:.4f}  macro_f1={f1:.4f}  comp={comp:.4f}")
    with open(output_dir / "eval_test_sampled.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(output_dir / "train_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
