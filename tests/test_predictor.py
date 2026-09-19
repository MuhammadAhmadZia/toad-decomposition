"""The predictor must match LightGBM exactly, or every post-hoc quality number
in the paper is wrong. This is the first thing to run after any change.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from toaddecomp.trees import tree_roots, predict_raw
from toaddecomp.accounting import bits_width, reference_bits, succinct_bits, kb
from toaddecomp.coding import cluster_leaves, quantize_leaves


def _model(depth=4, leaves=16, n_est=60):
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    m = lgb.LGBMClassifier(num_leaves=leaves, max_depth=depth, n_estimators=n_est,
                           verbose=-1, random_state=1).fit(Xtr, ytr)
    return m, Xte, yte


def test_predictor_matches_lightgbm():
    m, Xte, _ = _model()
    roots, _ = tree_roots(m.booster_)
    err = np.abs(m.booster_.predict(Xte, raw_score=True) - predict_raw(roots, Xte)).max()
    assert err < 1e-9, f"predictor disagrees with LightGBM by {err}"


def test_bits_width_matches_reference_helper():
    assert [bits_width(n) for n in (0, 1, 2, 3, 4, 7, 8)] == [1, 1, 2, 2, 3, 3, 4]


def test_sixteen_bit_quantization_is_lossless():
    m, Xte, _ = _model()
    roots, _ = tree_roots(m.booster_)
    base = predict_raw(roots, Xte)
    q16 = predict_raw(quantize_leaves(roots, 16), Xte)
    assert np.abs(base - q16).max() < 1e-3


def test_succinct_layout_wins_only_at_depth():
    """Below depth 4 the bitmap costs more than the padding it removes."""
    shallow, Xte, _ = _model(depth=2, leaves=4)
    deep, _, _ = _model(depth=8, leaves=256, n_est=120)
    rs, nf = tree_roots(shallow.booster_)
    rd, _ = tree_roots(deep.booster_)
    assert reference_bits(rs, nf, 2) / succinct_bits(rs, nf) < 1.0
    assert reference_bits(rd, nf, 8) / succinct_bits(rd, nf) > 1.5


def test_leaf_clustering_reduces_size():
    m, Xte, _ = _model()
    roots, nf = tree_roots(m.booster_)
    assert reference_bits(cluster_leaves(roots, 8), nf, 4) < reference_bits(roots, nf, 4)
