"""Memory accounting for compact tree-ensemble layouts.

Two layouts are priced.

``reference_bits`` reimplements the layout of Herrmann et al. (ICLR 2026) from
their published source, ``src/treelearner/memory_restricted_forest.hpp`` in
github.com/TinyAIoT/LightGBM-ToaD. Two details of that implementation matter
and are easy to miss. Tree reference arrays are sized from the *configured*
maximum depth rather than the depth a tree actually reaches, so a shallow tree
in a deep configuration still pays for the whole array. Threshold indices are
charged per real internal node, at the width that node's own feature needs.

``succinct_bits`` keeps every non-tree term identical and replaces the implicit
complete-tree array with a level-order (LOUDS) bitmap plus rank support, with
payload arrays addressed by rank and split by node type. It is lossless:
predictions are bit-identical either way, only the accounting changes.

See docs/ASSUMPTIONS.md for what was read from source and what remains a
modelling choice.
"""
from __future__ import annotations
import math
import numpy as np

from .trees import collect, is_leaf, n_internal_leaves, tree_depth, count_nodes

__all__ = ["bits_width", "threshold_width", "reference_bits", "succinct_bits",
           "dense_bits", "padding_ratio", "kb"]

RANK_OVERHEAD_DEFAULT = 0.25   # space for rank/select support over the bitmap


def bits_width(n: int) -> int:
    """Their bits() helper: bits(0) = 1, bits(n) = floor(log2(n)) + 1."""
    n = int(n)
    if n <= 0:
        return 1
    b = 0
    while n:
        b += 1
        n >>= 1
    return b


def threshold_width(values):
    """Narrowest representation for a set of thresholds, per their Sec. 3.2.1.

    Returns (bit_width, "int" | "float").
    """
    a = np.asarray(sorted(set(values)), dtype=np.float64)
    if a.size == 0:
        return 1, "int"
    if set(a.tolist()) <= {0.0, 1.0}:
        return 1, "int"
    if np.all(a == np.round(a)):
        lo, hi = a.min(), a.max()
        for w in (2, 4, 8, 16, 32):
            if lo >= 0 and hi <= 2 ** w - 1:
                return w, "int"
            if lo >= -(2 ** (w - 1)) and hi <= 2 ** (w - 1) - 1:
                return w, "int"
        return 32, "int"
    if np.array_equal(a.astype(np.float16).astype(np.float64), a):
        return 16, "float"
    return 32, "float"


def _shared_terms(roots, n_input_features, float_threshold_bits, leaf_value_bits):
    feats, thr, leaves = collect(roots)
    n_feat, n_leaf_vals = len(feats), len(leaves)
    max_thr = max((len(v) for v in thr.values()), default=1)

    mapping = n_feat * (4 + bits_width(n_input_features - 1) + bits_width(max_thr))
    threshold_table = sum(
        len(thr[f]) * (1 if threshold_width(thr[f])[0] == 1 else float_threshold_bits)
        for f in feats)
    leaf_table = n_leaf_vals * leaf_value_bits + (64 if leaf_value_bits < 32 else 0)
    return thr, n_feat, n_leaf_vals, max_thr, mapping + threshold_table + leaf_table


def reference_bits(roots, n_input_features, max_depth_cfg,
                   float_threshold_bits: int = 32, leaf_value_bits: int = 32) -> int:
    """Total bits under the reference layout."""
    thr, n_feat, n_leaf_vals, max_thr, shared = _shared_terms(
        roots, n_input_features, float_threshold_bits, leaf_value_bits)

    leaf_slots = 2 ** max_depth_cfg
    node_slots = 2 ** max_depth_cfg - 1
    leaf_ref = bits_width(n_leaf_vals - 1)
    feat_ref = bits_width(n_feat - 1)

    tree_bits = 0
    for r in roots:
        tree_bits += leaf_ref * leaf_slots + feat_ref * node_slots
        stack = [r]
        while stack:
            node = stack.pop()
            if not is_leaf(node):
                size = max(len(thr[node["split_feature"]]) - 1, 1)
                tree_bits += bits_width(size)
                stack.append(node["left_child"])
                stack.append(node["right_child"])
    return shared + tree_bits


def succinct_bits(roots, n_input_features,
                  rank_overhead: float = RANK_OVERHEAD_DEFAULT,
                  float_threshold_bits: int = 32, leaf_value_bits: int = 32) -> int:
    """Total bits under the succinct (LOUDS) layout. Lossless."""
    thr, n_feat, n_leaf_vals, max_thr, shared = _shared_terms(
        roots, n_input_features, float_threshold_bits, leaf_value_bits)

    leaf_ref = bits_width(n_leaf_vals - 1)
    feat_ref = bits_width(n_feat - 1)

    tree_bits = 0
    for r in roots:
        n_int, n_leaf = n_internal_leaves(r)
        tree_bits += int(math.ceil(2 * (n_int + n_leaf) * (1 + rank_overhead)))
        tree_bits += n_leaf * leaf_ref
        stack = [r]
        while stack:
            node = stack.pop()
            if not is_leaf(node):
                size = max(len(thr[node["split_feature"]]) - 1, 1)
                tree_bits += feat_ref + bits_width(size)
                stack.append(node["left_child"])
                stack.append(node["right_child"])
    return shared + tree_bits


def dense_bits(roots, bits_per_node: int = 128) -> int:
    """Conventional pointer-based array: 128 bits/node float32, 64 for fp16."""
    return sum(count_nodes(r) for r in roots) * bits_per_node


def padding_ratio(roots, max_depth_cfg) -> float:
    """Slots charged per real node under the configured-depth accounting."""
    real = sum(sum(n_internal_leaves(r)) for r in roots)
    slots = len(roots) * (2 ** (max_depth_cfg + 1) - 1)
    return slots / real if real else float("nan")


def kb(bits: float) -> float:
    return bits / 8 / 1024
