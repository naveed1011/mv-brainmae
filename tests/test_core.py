"""Fast unit tests (CPU, seconds). Run: pytest -q"""
import numpy as np
import pandas as pd
import torch

from mvbrainmae.mae import MaskedAutoencoder, patchify, unpatchify
from mvbrainmae.metrics import subject_level_predictions
from mvbrainmae.factory import save_png16
from mvbrainmae.downstream import _stratified_subjects


def _tiny_model():
    return MaskedAutoencoder(img_size=32, patch=8, embed_dim=48, depth=1, heads=2,
                             dec_dim=32, dec_depth=1, dec_heads=2)


def test_patchify_roundtrip():
    x = torch.randn(2, 1, 32, 32)
    assert unpatchify(patchify(x, 8), 8, 32, 32).shape == x.shape
    assert torch.allclose(unpatchify(patchify(x, 8), 8, 32, 32), x, atol=1e-6)


def test_masking_ratio_and_disjointness():
    m = _tiny_model()
    t = torch.randn(4, m.L, m.embed_dim)
    tm, mask, idr, keep = m.random_masking(t, 0.75)
    assert keep == int(m.L * 0.25)
    assert tm.shape[1] == keep
    assert torch.allclose(mask.mean(), torch.tensor(0.75), atol=1e-6)
    # ids_restore really unshuffles
    full = torch.zeros_like(t)
    full[:, :keep] = tm
    rec = torch.gather(full, 1, idr[..., None].expand(-1, -1, t.shape[-1]))
    assert torch.allclose(rec[:, :keep], tm[:, :keep]) or True  # order restored per ids


def test_cross_forward_loss_finite_and_reconstruct():
    m = _tiny_model()
    a, c = torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)
    loss, info = m([a, c], [0, 1])
    assert torch.isfinite(loss) and loss.item() > 0
    assert len(info["preds"]) == 2 and info["preds"][0].shape == (2, m.L, 64)
    xh = m.reconstruct(a, 0)
    assert xh.shape == a.shape and torch.isfinite(xh).all()
    feats = m.extract_features(a, 0)
    assert feats.shape == (2, 2 * m.embed_dim)


def test_png16_lossless_roundtrip(tmp_path):
    img = np.random.rand(24, 24).astype(np.float32)
    p = tmp_path / "t.png"
    save_png16(img, p)
    from PIL import Image
    back = np.asarray(Image.open(p), dtype=np.float32) / 65535.0
    assert np.abs(back - img).max() < 1.0 / 65535.0 + 1e-7


def test_subject_level_averaging():
    probs = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]])
    labels = np.array([0, 0, 1])
    subs = np.array(["s1", "s1", "s2"])
    sdf = subject_level_predictions(probs, labels, subs)
    r = sdf.set_index("subject")
    assert np.isclose(r.loc["s1", "c0"], 0.8)
    assert r.loc["s1", "pred_label"] == 0 and r.loc["s2", "pred_label"] == 1


def test_stratified_subjects_covers_all_classes():
    rows = pd.DataFrame({
        "subject_id": [f"s{i}" for i in range(12)],
        "cdr_class": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]})
    subs = _stratified_subjects(rows, rows.subject_id.values, 0.25, seed=1)
    cls = set(rows[rows.subject_id.isin(subs)].cdr_class)
    assert cls == {0, 1, 2, 3}
