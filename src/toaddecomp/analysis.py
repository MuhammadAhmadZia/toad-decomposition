"""Turning the grid into the numbers that appear in the paper.

Compression is read horizontally: for a fixed quality level, find the smallest
model each arm needs. Each learner is priced against its own float32 baseline,
because pricing CatBoost against a LightGBM baseline would fold a difference in
learner quality into a figure labelled compression.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["BUDGETS", "TECHNIQUES", "min_memory_for", "compression_table",
           "cross_learner_table", "depth_table", "sensitivity_table", "envelope"]

BUDGETS = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128])

TECHNIQUES = [
    ("16-bit baseline",            "lgb",        "fp16_kb"),
    ("Reference layout",           "lgb",        "reference_kb"),
    ("Succinct layout",            "lgb",        "succinct_kb"),
    ("Layout + leaf clustering",   "lgb+leaf",   "reference_kb"),
    ("Succinct + leaf clustering", "lgb+leaf",   "succinct_kb"),
    ("Succinct + 8-bit leaves",    "lgb+quant",  "succinct_kb"),
]


def min_memory_for(df: pd.DataFrame, column: str, level: float) -> float:
    """Smallest mean size among configurations reaching this quality level."""
    if not len(df):
        return np.nan
    g = df.groupby("config", as_index=False).agg({column: "mean", "metric": "mean"})
    reached = g[g.metric >= level]
    return reached[column].min() if len(reached) else np.nan


def _levels(df: pd.DataFrame, n: int = 5):
    lo, hi = df.metric.quantile(0.25), df.metric.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return []
    return np.linspace(lo, hi * 0.999, n)


def compression_table(grid: pd.DataFrame) -> pd.DataFrame:
    """Table II of the paper. Each learner against its own float32 baseline."""
    rows = []
    for ds in grid.dataset.unique():
        d = grid[grid.dataset == ds]
        base = d[d.arm == "lgb"]
        for level in _levels(base):
            ref = min_memory_for(base, "fp32_kb", level)
            for label, arm, column in TECHNIQUES:
                m = min_memory_for(d[d.arm == arm], column, level)
                rows.append(dict(dataset=ds, technique=label, level=level,
                                 ratio=ref / m if m and np.isfinite(m) and m > 0 else np.nan))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot_table(index="technique", columns="dataset",
                                          values="ratio", aggfunc="mean")


def cross_learner_table(grid: pd.DataFrame) -> pd.DataFrame:
    """Model selection, not compression. Reported separately and labelled so."""
    rows = []
    for ds in grid.dataset.unique():
        d = grid[grid.dataset == ds]
        lgb_arm, cb_arm = d[d.arm == "lgb"], d[d.arm == "catboost"]
        if not len(lgb_arm) or not len(cb_arm):
            continue
        lo = max(lgb_arm.metric.quantile(0.25), cb_arm.metric.quantile(0.25))
        hi = min(lgb_arm.metric.max(), cb_arm.metric.max())
        if hi <= lo:
            continue
        for level in np.linspace(lo, hi * 0.999, 5):
            best_lgb = min(min_memory_for(d[d.arm == a], "succinct_kb", level)
                           for a in ("lgb", "lgb+leaf", "lgb+quant"))
            best_cb = min_memory_for(d[d.arm == "catboost_oblivious"], "reference_kb", level)
            rows.append(dict(dataset=ds, level=level,
                             catboost_advantage=best_lgb / best_cb
                             if best_cb and np.isfinite(best_cb) and best_cb > 0 else np.nan))
    if not rows:
        return pd.DataFrame(columns=["catboost_advantage"])
    return pd.DataFrame(rows).groupby("dataset").catboost_advantage.mean().to_frame()


def depth_table(grid: pd.DataFrame) -> pd.DataFrame:
    """Table III. Padding, size reduction and loss rate by configured depth."""
    d = grid[grid.arm == "lgb"].copy()
    d["gain"] = d.reference_kb / d.succinct_kb
    return (d.groupby("depth")
             .agg(padding=("padding", "mean"),
                  reduction=("gain", "mean"),
                  loses_pct=("gain", lambda s: 100 * (s < 1).mean()))
             .round(3))


def sensitivity_table(datasets, seeds, struct, n_est=60):
    """Vary the accounting choices that are modelling decisions, not facts.

    The rank-support overhead, the width assumed for float thresholds and the
    rounding convention are all judgement calls. What must survive is the
    ordering, not the exact figures.
    """
    import lightgbm as lgb
    from .accounting import reference_bits, succinct_bits
    from .data import split
    from .trees import tree_roots

    settings = [("rank overhead 0%", dict(rank_overhead=0.0)),
                ("rank overhead 25%", dict(rank_overhead=0.25)),
                ("rank overhead 50%", dict(rank_overhead=0.5)),
                ("rank overhead 100%", dict(rank_overhead=1.0))]
    rows = []
    for ds, (task, X, y) in datasets.items():
        Xtr, Xte, ytr, yte = split(task, X, y, seeds[0])
        Model = lgb.LGBMClassifier if task == "clf" else lgb.LGBMRegressor
        n_feat = Xtr.shape[1]
        for n_leaves, depth in struct:
            m = Model(num_leaves=n_leaves, max_depth=depth, n_estimators=n_est,
                      max_bin=255, verbose=-1, random_state=seeds[0], n_jobs=-1).fit(Xtr, ytr)
            roots, _ = tree_roots(m.booster_)
            base = reference_bits(roots, n_feat, depth)
            for label, kw in settings:
                rows.append(dict(dataset=ds, depth=depth, setting=label,
                                 gain=base / succinct_bits(roots, n_feat, **kw)))
            rows.append(dict(
                dataset=ds, depth=depth, setting="float thresholds 16-bit",
                gain=reference_bits(roots, n_feat, depth, float_threshold_bits=16) /
                     succinct_bits(roots, n_feat, float_threshold_bits=16)))
    return pd.DataFrame(rows).pivot_table(index="depth", columns="setting", values="gain")


def envelope(sub: pd.DataFrame, column: str) -> np.ndarray:
    """Best mean quality reachable inside each budget. Seeds averaged first."""
    g = sub.groupby("config", as_index=False).agg({column: "mean", "metric": "mean"})
    return np.array([g.loc[g[column] <= b, "metric"].max() if (g[column] <= b).any() else np.nan
                     for b in BUDGETS])
