"""Downstream heads.

LinearProbe        : logistic regression on frozen [CLS||mean] features (torch impl,
                     so the exact same features can also feed sklearn — see downstream.py)
AttentionPoolHead  : single-query attention over patch tokens + MLP (for fine-tuning)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttentionPoolHead(nn.Module):
    def __init__(self, dim: int, n_classes: int, hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L, D) patch tokens (CLS excluded)."""
        attn = torch.softmax((self.query @ tokens.transpose(1, 2)) * self.scale,
                             dim=-1)                     # (B, 1, L)
        pooled = (attn @ tokens).squeeze(1)              # (B, D)
        return self.mlp(self.norm(pooled))


class TimmEncoderAdapter(nn.Module):
    """Adapts a timm ViT (no num_classes) to the forward_tokens(x, view_id) API,
    replicating grayscale input to 3 channels. view_id accepted & ignored —
    the baseline arms have no view awareness (by design)."""

    def __init__(self, timm_model: nn.Module, dim: int = 384):
        super().__init__()
        self.model = timm_model
        self.dim = dim

    def forward_tokens(self, x: torch.Tensor, view_id: int = 0) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model.forward_features(x)  # (B, 1+L, D)
