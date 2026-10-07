#!/usr/bin/env bash
# Push the run-all notebook to your Kaggle account as a GPU notebook.
# Prereq: pip install kaggle && ~/.kaggle/kaggle.json API token
# Edit scripts/kaggle/kernel-metadata.json first (username + dataset id).
set -euo pipefail
cd "$(dirname "$0")"
cp ../notebooks/run_all_kaggle.ipynb kaggle/run_all_kaggle.ipynb
kaggle kernels push -p kaggle
echo "Pushed. Open it on Kaggle, confirm GPU is on, and Run All."
