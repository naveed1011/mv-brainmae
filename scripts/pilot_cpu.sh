#!/usr/bin/env bash
# CPU pilot on REAL OASIS-1 data (laptop without GPU). ~1.5-3 h on 4 cores.
# Needs volumes under data/oasis1 (see docs/KAGGLE_SETUP.md §1).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d data/oasis1 ] || { echo "Put OASIS-1 volumes in data/oasis1 first (docs/KAGGLE_SETUP.md)"; exit 1; }
python -m mvbrainmae.factory   --config configs/pilot/factory.yaml
python -m mvbrainmae.pretrain  --config configs/pilot/pretrain_single.yaml
python -m mvbrainmae.pretrain  --config configs/pilot/pretrain_cross.yaml
python -m mvbrainmae.downstream --config configs/pilot/downstream.yaml
python -m mvbrainmae.anomaly   --config configs/pilot/anomaly.yaml
python -m mvbrainmae.analysis  --config configs/pilot/analysis.yaml
echo "PILOT COMPLETE — see results_pilot/summary.md (pipeline proof, not the science run)"
