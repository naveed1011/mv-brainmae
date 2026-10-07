"""The model we build: a Masked Autoencoder ViT with learned VIEW EMBEDDINGS,
supporting multi-view (cross-plane) masked pretraining.

MAE recipe follows He et al. 2022 (arXiv:2111.06377) — 75% random masking,
asymmetric encoder/decoder, normalized-pixel reconstruction targets. The
multi-view design (cross-plane context, learned view embeddings, symmetric
two-view reconstruction) is the original twist of this project.

Sequence layout (cross mode, batch B, L patches/view, keep = 25% of L):

    encoder input : [CLS] [kept axial tokens + pos + view0] [kept coronal tokens + pos + view1]
    decoder input : [full axial seq (kept latents + mask tokens, restored order) + dec_pos + view0]
                    [full coronal seq ... + view1]
    loss          : mean L2 over masked patches of BOTH views (normalized pixels)
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Patchify helpers
# --------------------------------------------------------------------------- #
def patchify(imgs: torch.Tensor, p: int) -> torch.Tensor:
    """(B,1,H,W) -> (B, L, p*p)  — MAE layout: patches row-major, pixels row-major."""
    b, c, h, w = imgs.shape
    assert h % p == 0 and w % p == 0
    gh, gw = h // p, w // p
    x = imgs.reshape(b, c, gh, p, gw, p)
    x = torch.einsum("bcghpw->bgphwc", x)
    return x.reshape(b, gh * gw, p * p * c)


def unpatchify(preds: torch.Tensor, p: int, h: int, w: int) -> torch.Tensor:
    """(B, L, p*p) -> (B,1,H,W)"""
    b = preds.shape[0]
    gh, gw = h // p, w // p
    x = preds.reshape(b, gh, gw, p, p, 1)
    x = torch.einsum("bgphwc->bcghpw", x)
    return x.reshape(b, 1, h, w)


def _block(dim: int, heads: int, mlp_ratio: float, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True)


class MaskedAutoencoder(nn.Module):
    def __init__(self, img_size: int = 224, patch: int = 16, in_chans: int = 1,
                 embed_dim: int = 384, depth: int = 6, heads: int = 6,
                 mlp_ratio: float = 4.0, dec_dim: int = 256, dec_depth: int = 2,
                 dec_heads: int = 4, mask_ratio: float = 0.75,
                 norm_pix: bool = True, n_views: int = 2, dropout: float = 0.0):
        super().__init__()
        assert img_size % patch == 0
        self.cfg_dict = dict(img_size=img_size, patch=patch, in_chans=in_chans,
                             embed_dim=embed_dim, depth=depth, heads=heads,
                             mlp_ratio=mlp_ratio, dec_dim=dec_dim,
                             dec_depth=dec_depth, dec_heads=dec_heads,
                             mask_ratio=mask_ratio, norm_pix=norm_pix,
                             n_views=n_views, dropout=dropout)
        self.p, self.img_size = patch, img_size
        self.embed_dim, self.depth = embed_dim, depth
        self.L = (img_size // patch) ** 2
        self.mask_ratio, self.norm_pix = mask_ratio, norm_pix

        # ---- encoder ----
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.L, embed_dim))
        self.view_embed = nn.Embedding(n_views, embed_dim)
        nn.init.zeros_(self.view_embed.weight)  # views start identical, diverge by learning
        self.blocks = nn.ModuleList(
            [_block(embed_dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # ---- decoder ----
        self.dec_proj = nn.Linear(embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.L, dec_dim))
        self.dec_view_embed = nn.Embedding(n_views, dec_dim)
        nn.init.zeros_(self.dec_view_embed.weight)
        self.dec_blocks = nn.ModuleList(
            [_block(dec_dim, dec_heads, mlp_ratio, dropout) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_head = nn.Linear(dec_dim, patch * patch * in_chans)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # Tokenization + masking
    # ------------------------------------------------------------------ #
    def _embed_tokens(self, x: torch.Tensor, view_id: int) -> torch.Tensor:
        t = self.patch_embed(x).flatten(2).transpose(1, 2)      # (B, L, D)
        vid = torch.tensor([view_id], device=x.device)
        return t + self.pos_embed + self.view_embed(vid)

    def random_masking(self, t: torch.Tensor, mask_ratio: float):
        b, l, d = t.shape
        keep = max(1, int(l * (1 - mask_ratio)))
        noise = torch.rand(b, l, device=t.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        t_masked = torch.gather(t, 1, ids_keep[..., None].expand(-1, -1, d))
        mask = torch.ones(b, l, device=t.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)  # unshuffle: 1 = removed
        return t_masked, mask, ids_restore, keep

    # ------------------------------------------------------------------ #
    # Encoder / decoder over a LIST of views
    # ------------------------------------------------------------------ #
    def encode_views(self, views: list[torch.Tensor], view_ids: list[int],
                     mask: bool = True):
        """Returns (latent [B, 1+sum(keep), D], masks list, ids_restore list, keep)."""
        toks, masks, idrs = [], [], []
        for x, vid in zip(views, view_ids):
            t = self._embed_tokens(x, vid)
            if mask:
                tm, m, idr, keep = self.random_masking(t, self.mask_ratio)
            else:
                tm, m, idr, keep = t, None, None, t.shape[1]
            toks.append(tm); masks.append(m); idrs.append(idr)
        cls = self.cls_token.expand(toks[0].shape[0], -1, -1)
        seq = torch.cat([cls] + toks, dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        return self.norm(seq), masks, idrs, keep

    def decode_views(self, latent: torch.Tensor, idrs: list, n_views_seq: list[int],
                     view_ids: list[int]) -> list[torch.Tensor]:
        """latent: (B, 1+sum(keep), D). Returns per-view preds (B, L, p*p)."""
        b = latent.shape[0]
        lat = self.dec_proj(latent[:, 1:])  # drop CLS; (B, sum_keep, dec_dim)
        chunks = torch.split(lat, n_views_seq, dim=1)
        seqs = []
        for chunk, idr, vid in zip(chunks, idrs, view_ids):
            keep = chunk.shape[1]
            mask_tokens = self.mask_token.expand(b, self.L - keep, -1)
            full = torch.cat([chunk, mask_tokens], dim=1)               # shuffled order
            full = torch.gather(                                       # unshuffle
                full, 1, idr[..., None].expand(-1, -1, full.shape[-1]))
            vemb = self.dec_view_embed(
                torch.tensor([vid], device=full.device)).unsqueeze(0)
            seqs.append(full + self.dec_pos_embed + vemb)
        seq = torch.cat(seqs, dim=1)
        for blk in self.dec_blocks:
            seq = blk(seq)
        seq = self.dec_norm(seq)
        preds = self.dec_head(seq)
        return list(torch.split(preds, self.L, dim=1))

    # ------------------------------------------------------------------ #
    # Training forward + loss
    # ------------------------------------------------------------------ #
    def forward_loss(self, views: list[torch.Tensor], view_ids: list[int],
                     preds: list[torch.Tensor], masks: list[torch.Tensor]):
        total, n_terms = 0.0, 0
        for x, pred, m in zip(views, preds, masks):
            target = patchify(x, self.p)                     # (B, L, p*p)
            if self.norm_pix:
                mu = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mu) / torch.sqrt(var + 1e-6)
            se = ((pred - target) ** 2).mean(dim=-1)         # (B, L)
            loss_v = (se * m).sum() / m.sum().clamp(min=1)   # masked mean
            total = total + loss_v; n_terms += 1
        return total / max(1, n_terms)

    def forward(self, views: list[torch.Tensor], view_ids: list[int]):
        latent, masks, idrs, keep = self.encode_views(views, view_ids, mask=True)
        n_seq = [keep] * len(views)
        preds = self.decode_views(latent, idrs, n_seq, view_ids)
        loss = self.forward_loss(views, view_ids, preds, masks)
        return loss, {"preds": preds, "masks": masks}

    # ------------------------------------------------------------------ #
    # Inference utilities (downstream features / anomaly reconstruction)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, view_id: int = 0) -> torch.Tensor:
        """Full-image encode (no masking) -> [CLS || mean-pool] (B, 2*embed_dim)."""
        latent, _, _, _ = self.encode_views([x], [view_id], mask=False)
        cls, toks = latent[:, 0], latent[:, 1:]
        return torch.cat([cls, toks.mean(dim=1)], dim=-1)

    def forward_tokens(self, x: torch.Tensor, view_id: int = 0) -> torch.Tensor:
        """Full-image encode -> (B, 1+L, D) including CLS (for fine-tuning heads).

        NOTE: intentionally NOT under no_grad — fine-tuning backprops through it.
        """
        latent, _, _, _ = self.encode_views([x], [view_id], mask=False)
        return latent

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, view_id: int = 0) -> torch.Tensor:
        """Full reconstruction (all patches visible) in DATASET-normalized space."""
        latent, _, idrs, keep = self.encode_views([x], [view_id], mask=False)
        # no masking happened -> ids_restore is None; build identity restore
        b = x.shape[0]
        idr = torch.arange(self.L, device=x.device).expand(b, -1)
        preds = self.decode_views(latent, [idr], [self.L], [view_id])[0]
        return unpatchify(preds, self.p, self.img_size, self.img_size)

    @classmethod
    def from_config(cls, cfg: dict) -> "MaskedAutoencoder":
        keys = ("img_size", "patch", "in_chans", "embed_dim", "depth", "heads",
                "mlp_ratio", "dec_dim", "dec_depth", "dec_heads", "mask_ratio",
                "norm_pix", "n_views", "dropout")
        return cls(**{k: cfg[k] for k in keys if k in cfg})


def build_from_checkpoint(ckpt_path: str, map_location="cpu") -> MaskedAutoencoder:
    """Rebuild a MaskedAutoencoder from an encoder-only or full checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    model = MaskedAutoencoder.from_config(ckpt.get("config", {}))
    model.load_state_dict(ckpt["model"], strict=False)  # encoder-only ckpts lack decoder
    model.eval()
    return model
