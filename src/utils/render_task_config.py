"""Render a task-specific config from an existing YAML template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def infer_num_labels(processed_dir: Path) -> int:
    labels = set()
    train_path = processed_dir / "train.jsonl"
    with train_path.open(encoding="utf-8") as handle:
        for line in handle:
            labels.add(json.loads(line)["label"])
    if not labels:
        raise ValueError(f"No labels found in {train_path}")
    return len(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    template_path = Path(args.template)
    output_path = Path(args.output)
    processed_dir = Path(args.processed_dir)
    num_labels = infer_num_labels(processed_dir)

    with template_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["project_name"] = args.project_name
    config["dataset"] = {
        "train_path": str(processed_dir / "train.jsonl"),
        "valid_path": str(processed_dir / "valid.jsonl"),
        "test_path": str(processed_dir / "test.jsonl"),
    }

    training = config.setdefault("training", {})
    if "output_dir" in training:
        training["output_dir"] = args.output_dir

    evaluation = config.setdefault("evaluation", {})
    if "output_dir" in evaluation:
        evaluation["output_dir"] = args.output_dir

    model = config.get("model")
    if isinstance(model, dict) and "num_labels" in model:
        model["num_labels"] = num_labels

    env = config.get("env")
    if isinstance(env, dict) and "num_labels" in env:
        env["num_labels"] = num_labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    print(f"Rendered {output_path} with num_labels={num_labels}")


if __name__ == "__main__":
    main()
