#!/usr/bin/env python3
"""
Monitor status of all review response experiments.
Tracks Lagrangian RL, Segment-PE ablation, and Lagrangian results.
"""

import json
from pathlib import Path
from datetime import datetime
import subprocess

def get_slurm_job_status(job_id):
    """Get SLURM job status and info."""
    try:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "-o", "%T|%P|%l|%i"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                parts = lines[1].split('|')
                return {
                    "status": parts[0],
                    "partition": parts[1],
                    "time_limit": parts[2],
                    "job_id": parts[3],
                }
    except:
        pass
    return None

def check_experiment_results(task):
    """Check if experiment results exist."""
    results = {}

    # Lagrangian
    lag_path = Path(f"results/rl_runs/grpo_finetuned_env_{task}_lagrangian/eval_test_sampled.json")
    if lag_path.exists():
        with open(lag_path) as f:
            metrics = json.load(f)
            results['lagrangian'] = {
                'f1': metrics.get('f1'),
                'accuracy': metrics.get('accuracy'),
                'compression': metrics.get('compression_ratio'),
            }

    # Segment-PE (if applicable)
    if task in ['promoter', 'splice_site']:
        seg_path = Path(f"results/rl_runs/grpo_finetuned_env_segment_pe_{task}/eval_test_sampled.json")
        if seg_path.exists():
            with open(seg_path) as f:
                metrics = json.load(f)
                results['segment_pe'] = {
                    'f1': metrics.get('f1'),
                    'accuracy': metrics.get('accuracy'),
                    'compression': metrics.get('compression_ratio'),
                }

    return results

def main():
    print("=" * 100)
    print("EXPERIMENT STATUS MONITOR")
    print("=" * 100)
    print()

    # Job information
    jobs = {
        34526906: {
            "name": "Lagrangian RL (all 7 GUE tasks)",
            "type": "RL training",
            "expected_runtime": "12-24 hours",
            "outputs": [
                "results/rl_runs/grpo_finetuned_env_promoter_lagrangian/",
                "results/rl_runs/grpo_finetuned_env_human_tf_0_lagrangian/",
                "results/rl_runs/grpo_finetuned_env_human_tf_1_lagrangian/",
                "results/rl_runs/grpo_finetuned_env_human_tf_2_lagrangian/",
                "results/rl_runs/grpo_finetuned_env_human_tf_3_lagrangian/",
                "results/rl_runs/grpo_finetuned_env_human_tf_4_lagrangian/",
                "results/rl_runs/grpo_finetuned_env_splice_site_lagrangian/",
            ]
        },
        34526982: {
            "name": "Segment-Aware PE Ablation (promoter + splice-site)",
            "type": "Ablation study",
            "expected_runtime": "4-8 hours",
            "outputs": [
                "results/rl_runs/grpo_finetuned_env_segment_pe_promoter/",
                "results/rl_runs/grpo_finetuned_env_segment_pe_splice_site/",
            ]
        },
    }

    print("ACTIVE JOBS:")
    print("-" * 100)

    for job_id, info in jobs.items():
        status_info = get_slurm_job_status(job_id)

        if status_info:
            status = status_info["status"]
            status_symbol = "🟢" if status == "RUNNING" else "⏳" if status == "PENDING" else "❌"
        else:
            status = "NOT FOUND"
            status_symbol = "❓"

        print(f"\n{status_symbol} Job {job_id}: {info['name']}")
        print(f"   Type: {info['type']}")
        print(f"   Expected runtime: {info['expected_runtime']}")

        if status_info:
            print(f"   Status: {status} on {status_info['partition']} partition")
        else:
            print(f"   Status: {status}")

    print("\n" + "=" * 100)
    print("EXPECTED RESULTS:")
    print("=" * 100)

    print("\n1. LAGRANGIAN RL (job 34526906) - When complete:")
    print("   - Full results across 7 tasks (promoter, TF0-4, splice-site)")
    print("   - Compression control at target_compression=0.18")
    print("   - Key question: Does Lagrangian fix over-compression on splice-site?")
    print("   - Add to paper: Table row showing Lagrangian RL results")

    print("\n2. SEGMENT-PE ABLATION (job 34526982) - When complete:")
    print("   - Promoter: Compare standard vs segment-PE RL")
    print("   - Splice-site: Compare standard vs segment-PE RL")
    print("   - Key metric: Does segment-PE improve boundary-sensitive tasks?")
    print("   - Add to paper: Ablation section showing segment-PE benefit")

    print("\n3. MOTIF SIGNIFICANCE (completed):")
    print("   - Permutation tests for TATA enrichment")
    print("   - Statistical p-values with effect sizes")
    print("   - Controls for positional bias and co-occurrence")
    print("   - Add to paper: Appendix with statistical significance")

    print("\n" + "=" * 100)
    print("PROGRESS SUMMARY:")
    print("=" * 100)

    print("\n✅ COMPLETED:")
    print("   - Compression-performance curves (7 figures)")
    print("   - Motif significance analysis (permutation tests)")
    print("   - Lagrangian configs for all 7 tasks")
    print("   - Segment-PE environment + configs")
    print("   - Training/eval script updates")

    print("\n⏳ IN PROGRESS:")
    print("   - Job 34526906: Lagrangian RL (all 7 tasks)")
    print("   - Job 34526982: Segment-PE ablation (2 tasks)")

    print("\n📝 NEXT STEPS (when jobs complete):")
    print("   1. Collect Lagrangian results across all 7 tasks")
    print("   2. Generate comparison table: Lagrangian vs stride-pooling")
    print("   3. Analyze segment-PE ablation benefit")
    print("   4. Revise paper with new results + figures")
    print("   5. Update response letter addressing all reviewer questions")

    print("\n" + "=" * 100)

    # Check for existing results
    print("\nCURRENT RESULTS SUMMARY:")
    print("-" * 100)

    tasks = ["promoter", "human_tf_0", "human_tf_1", "human_tf_2", "human_tf_3", "human_tf_4", "splice_site"]

    for task in tasks:
        results = check_experiment_results(task)
        if results:
            print(f"\n{task.upper()}:")
            for method, metrics in results.items():
                if metrics.get('f1'):
                    print(f"  {method:15s}: F1={metrics['f1']:.4f}, Comp={metrics.get('compression', 0):.4f}")

if __name__ == "__main__":
    main()
