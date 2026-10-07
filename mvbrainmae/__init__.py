"""MV-BrainMAE: Multi-View Masked Autoencoder foundation model for brain MRI.

An original experiment by Naveed Ahmad.

Modules
-------
oasis      : readers for OASIS-1-style volume collections + metadata CSVs
factory    : raw volumes -> QC'd 16-bit PNG slices + metadata/splits/folds
datasets   : PyTorch datasets for pretraining (single/cross/adjacent) and downstream
mae        : Masked Autoencoder ViT with learned view embeddings (the model we build)
heads      : linear probe + attention-pool classifier heads
pretrain   : stage 2 — masked pretraining loop (AMP, resume, time-guarded)
downstream : stage 3 — feature caching, GroupKFold probes, label efficiency, finetuning
anomaly    : stage 4 — reconstruction-error anomaly maps + clinical correlations
analysis   : stage 5 — figures + hypothesis verdicts (summary.md)
utils      : seeds, configs, checkpointing, logging
metrics    : subject-level evaluation metrics
"""

__version__ = "0.1.0"
__author__ = "Naveed Ahmad"
