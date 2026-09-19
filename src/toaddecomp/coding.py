"""Post-hoc value coding: clustering and quantisation of a fitted ensemble.

Every function here edits a model that already exists and can move predictions,
so each is scored on quality as well as size. Nothing retrains.
"""
from __future__ import annotations
import copy
import numpy as np

from .trees import collect, is_leaf

__all__ = ["cluster_thresholds", "cluster_leaves", "quantize_leaves", "apply_coding"]


def _kmeans_1d(values: np.ndarray, k: int, iters: int = 100) -> np.ndarray:
    kk = min(k, len(values))
    if kk >= len(values):
        return values
    centres = np.percentile(values, np.linspace(0, 100, kk))
    for _ in range(iters):
        assign = np.abs(values[:, None] - centres[None, :]).argmin(1)
        new = np.array([values[assign == j].mean() if (assign == j).any() else centres[j]
                        for j in range(kk)])
        if np.allclose(new, centres):
            break
        centres = new
    return centres


def cluster_thresholds(roots, k: int):
    """Snap each feature's thresholds to k centres. Analogue of a reuse penalty."""
    roots = copy.deepcopy(roots)
    _, thr, _ = collect(roots)
    mapping = {}
    for f, vals in thr.items():
        v = np.array(sorted(vals))
        c = _kmeans_1d(v, k)
        mapping[f] = {float(x): float(c[np.abs(c - x).argmin()]) for x in v}
    for r in roots:
        stack = [r]
        while stack:
            n = stack.pop()
            if not is_leaf(n):
                n["threshold"] = mapping[n["split_feature"]][float(n["threshold"])]
                stack.append(n["left_child"])
                stack.append(n["right_child"])
    return roots


def cluster_leaves(roots, k: int):
    """Snap all leaf values to k global centres."""
    roots = copy.deepcopy(roots)
    _, _, leaves = collect(roots)
    v = np.array(sorted(leaves))
    c = _kmeans_1d(v, k)
    if len(c) >= len(v):
        return roots
    for r in roots:
        stack = [r]
        while stack:
            n = stack.pop()
            if is_leaf(n):
                x = float(np.float32(n["leaf_value"]))
                n["leaf_value"] = float(c[np.abs(c - x).argmin()])
            else:
                stack.append(n["left_child"])
                stack.append(n["right_child"])
    return roots


def quantize_leaves(roots, bits: int):
    """Uniform b-bit fixed point over the leaf range, shared scale and offset.

    Safe at 16 and 8 bits on every benchmark tested. At 4 bits it holds for
    classification and fails for regression, where leaf values span a far wider
    range than log-odds. See docs/RESULTS.md.
    """
    roots = copy.deepcopy(roots)
    _, _, leaves = collect(roots)
    v = np.array(sorted(leaves))
    lo, hi = v.min(), v.max()
    levels = 2 ** bits - 1
    for r in roots:
        stack = [r]
        while stack:
            n = stack.pop()
            if is_leaf(n):
                x = float(n["leaf_value"])
                q = round((x - lo) / (hi - lo) * levels) if hi > lo else 0
                n["leaf_value"] = lo + q * (hi - lo) / levels
            else:
                stack.append(n["left_child"])
                stack.append(n["right_child"])
    return roots


def apply_coding(roots, kind: str, k: int):
    if kind == "thr":
        return cluster_thresholds(roots, k)
    if kind == "leaf":
        return cluster_leaves(roots, k)
    if kind == "both":
        return cluster_thresholds(cluster_leaves(roots, k), k)
    if kind == "quant":
        return quantize_leaves(roots, k)
    raise ValueError(f"unknown coding: {kind}")
