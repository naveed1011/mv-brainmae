#!/usr/bin/env bash
# Full-pipeline smoke test on SYNTHETIC brains (CPU, ~10 min).
# The synthetic data carries a planted atrophy signal: a correct pipeline MUST
# detect it (H0 gate). If this fails, the CODE is broken — fix before Kaggle.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== synthetic data (planted signal) =="
python scripts/make_synthetic_data.py --out data/synthetic --n 32

echo "== stage 1: factory =="
python -m mvbrainmae.factory --config configs/smoke/factory.yaml

echo "== stage 2A: pretrain single =="
python -m mvbrainmae.pretrain --config configs/smoke/pretrain_single.yaml

echo "== stage 2B: pretrain cross =="
python -m mvbrainmae.pretrain --config configs/smoke/pretrain_cross.yaml

echo "== stage 3: downstream =="
python -m mvbrainmae.downstream --config configs/smoke/downstream.yaml

echo "== stage 4: anomaly =="
python -m mvbrainmae.anomaly --config configs/smoke/anomaly.yaml

echo "== stage 5: analysis =="
python -m mvbrainmae.analysis --config configs/smoke/analysis.yaml

echo "== H0 validity gate =="
if grep -q "H0.*PASS" results_smoke/summary.md; then
    echo "SMOKE TEST PASSED — pipeline detects the planted synthetic signal."
else
    echo "SMOKE TEST FAILED — H0 gate not passed; inspect results_smoke/summary.md"
    exit 1
fi
