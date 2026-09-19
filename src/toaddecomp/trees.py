"""Parsing, traversal and size statistics for LightGBM tree ensembles.

The vectorised predictor exists so that a model edited after fitting (clustered
thresholds, clustered or quantised leaves) can still be scored. LightGBM has no
API for predicting from a modified dump, so traversal is done here and checked
against the library's own output in tests/test_predictor.py.
"""
from __future__ import annotations
import numpy as np

__all__ = ["is_leaf", "tree_roots", "tree_depth", "count_nodes", "collect",
           "n_internal_leaves", "flatten_tree", "predict_raw"]


def is_leaf(node) -> bool:
    return "leaf_value" in node and "split_feature" not in node


def tree_roots(booster):
    """Return (list of tree-root dicts, number of input features)."""
    d = booster.dump_model()
    return [t["tree_structure"] for t in d["tree_info"]], d["max_feature_idx"] + 1


def tree_depth(node, d: int = 0) -> int:
    if is_leaf(node):
        return d
    return max(tree_depth(node["left_child"], d + 1),
               tree_depth(node["right_child"], d + 1))


def count_nodes(node) -> int:
    if is_leaf(node):
        return 1
    return 1 + count_nodes(node["left_child"]) + count_nodes(node["right_child"])


def n_internal_leaves(root):
    ni = nl = 0
    stack = [root]
    while stack:
        n = stack.pop()
        if is_leaf(n):
            nl += 1
        else:
            ni += 1
            stack.append(n["left_child"])
            stack.append(n["right_child"])
    return ni, nl


def collect(roots):
    """Distinct used features, thresholds per feature, distinct leaf values.

    Leaf values are de-duplicated at float32 precision, matching the global
    leaf table of the reference layout.
    """
    feats, thr, leaves = set(), {}, set()
    for r in roots:
        stack = [r]
        while stack:
            n = stack.pop()
            if is_leaf(n):
                leaves.add(float(np.float32(n["leaf_value"])))
            else:
                f = n["split_feature"]
                feats.add(f)
                thr.setdefault(f, set()).add(float(n["threshold"]))
                stack.append(n["left_child"])
                stack.append(n["right_child"])
    return feats, thr, leaves


def flatten_tree(root):
    """Dict tree to flat arrays, so traversal can run on all rows at once."""
    feat, thr, left, right, val = [], [], [], [], []

    def add(node):
        i = len(feat)
        feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); val.append(0.0)
        if is_leaf(node):
            val[i] = float(node["leaf_value"])
            return i
        feat[i] = int(node["split_feature"])
        thr[i] = float(node["threshold"])
        left[i] = add(node["left_child"])
        right[i] = add(node["right_child"])
        return i

    add(root)
    return (np.asarray(feat, np.int64), np.asarray(thr, np.float64),
            np.asarray(left, np.int64), np.asarray(right, np.int64),
            np.asarray(val, np.float64))


def predict_raw(roots, X) -> np.ndarray:
    """Raw scores. Apply a sigmoid yourself for binary classification."""
    X = np.ascontiguousarray(np.asarray(X, np.float64))
    n = X.shape[0]
    out = np.zeros(n)
    for feat, thr, left, right, val in (flatten_tree(r) for r in roots):
        idx = np.zeros(n, np.int64)
        for _ in range(64):
            active = np.flatnonzero(feat[idx] >= 0)
            if active.size == 0:
                break
            cur = idx[active]
            go_left = X[active, feat[cur]] <= thr[cur]
            idx[active] = np.where(go_left, left[cur], right[cur])
        out += val[idx]
    return out
