#!/usr/bin/env bash

set -e

echo "Running DNABERT-2 BPE baseline..."
python -m src.eval.evaluate_baseline --config configs/baseline.yaml
echo "Baseline complete."
