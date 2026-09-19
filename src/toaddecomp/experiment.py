"""The grid. Every cell is checkpointed, so an interrupted run resumes."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from .accounting import reference_bits, succinct_bits, dense_bits, padding_ratio, kb
from .coding import apply_coding
from .data import split
from .metrics import score
from .trees import tree_roots, tree_depth, predict_raw

__all__ = ["STRUCT", "N_EST", "MAX_BIN", "CODING_CFGS", "CB_DEPTH", "CB_ITERS",
           "run_lightgbm_grid", "run_catboost_grid"]

# (num_leaves, max_depth). Depth runs 1..8 so the settings used by
# Herrmann et al. (1, 2, 4, 8) are all covered.
STRUCT = [(2, 1), (4, 2), (8, 3), (16, 4), (32, 5), (64, 6), (128, 7), (256, 8)]
N_EST = [10, 30, 60, 120]
MAX_BIN = [255, 63, 15]

# post-hoc value coding, applied to the full-bin models only
CODING_CFGS = ([("leaf", k) for k in (256, 64, 16, 8, 4)] +
               [("thr", k) for k in (16, 8, 4)] +
               [("both", k) for k in (16, 8, 4)] +
               [("quant", b) for b in (16, 8, 4)])

CB_DEPTH = [1, 2, 3, 4, 5]
CB_ITERS = [10, 30, 60, 120]


def _ckpt(ckpt_dir: str, tag: str) -> str:
    return os.path.join(ckpt_dir, f"{tag}.csv")


def run_lightgbm_grid(datasets, seeds, out_dir="results", ckpt_dir="checkpoints",
                      verbose=True) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    parts = []

    for ds, (task, X, y) in datasets.items():
        for seed in seeds:
            tag = f"lgb_{ds}_s{seed}"
            if os.path.exists(_ckpt(ckpt_dir, tag)):
                parts.append(pd.read_csv(_ckpt(ckpt_dir, tag)))
                if verbose:
                    print(f"  {ds} seed={seed}: from checkpoint")
                continue

            rows = []
            Xtr, Xte, ytr, yte = split(task, X, y, seed)
            Model = lgb.LGBMClassifier if task == "clf" else lgb.LGBMRegressor
            n_feat = Xtr.shape[1]

            for n_leaves, depth in STRUCT:
                for n_est in N_EST:
                    for max_bin in MAX_BIN:
                        m = Model(num_leaves=n_leaves, max_depth=depth,
                                  n_estimators=n_est, max_bin=max_bin,
                                  verbose=-1, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
                        roots, _ = tree_roots(m.booster_)
                        rows.append(dict(
                            dataset=ds, seed=seed, arm="lgb", depth=depth,
                            config=f"nl{n_leaves}_d{depth}_n{n_est}_b{max_bin}",
                            reference_kb=kb(reference_bits(roots, n_feat, depth)),
                            succinct_kb=kb(succinct_bits(roots, n_feat)),
                            padding=padding_ratio(roots, depth),
                            fp32_kb=kb(dense_bits(roots, 128)),
                            fp16_kb=kb(dense_bits(roots, 64)),
                            **score(task, yte, m.booster_.predict(Xte, raw_score=True))))

                        if max_bin != 255:
                            continue
                        for kind, k in CODING_CFGS:
                            edited = apply_coding(roots, kind, k)
                            leaf_bits = k if kind == "quant" else 32
                            rows.append(dict(
                                dataset=ds, seed=seed, arm=f"lgb+{kind}", depth=depth,
                                config=f"nl{n_leaves}_d{depth}_n{n_est}_{kind}{k}",
                                reference_kb=kb(reference_bits(edited, n_feat, depth,
                                                               leaf_value_bits=leaf_bits)),
                                succinct_kb=kb(succinct_bits(edited, n_feat,
                                                             leaf_value_bits=leaf_bits)),
                                padding=padding_ratio(edited, depth),
                                fp32_kb=kb(dense_bits(edited, 128)),
                                fp16_kb=kb(dense_bits(edited, 64)),
                                **score(task, yte, predict_raw(edited, Xte))))

            part = pd.DataFrame(rows)
            part.to_csv(_ckpt(ckpt_dir, tag), index=False)
            parts.append(part)
            if verbose:
                print(f"  {ds} seed={seed}: {len(part)} rows, checkpointed")

    grid = pd.concat(parts, ignore_index=True)
    grid.to_csv(os.path.join(out_dir, "grid_lightgbm.csv"), index=False)
    return grid


def catboost_bits(cb_json, n_input_features, oblivious=False, leaf_value_bits=32):
    """Oblivious trees are complete by construction, so no padding is charged.

    ``oblivious=True`` additionally exploits level sharing: one (feature,
    threshold) pair per level rather than per node.
    """
    from .accounting import bits_width, threshold_width
    trees = cb_json.get("oblivious_trees", [])
    feats, thr, leaves, depths = set(), {}, set(), []
    for t in trees:
        splits = t.get("splits", [])
        depths.append(len(splits))
        for s in splits:
            f = s.get("float_feature_index")
            if f is None:
                continue
            feats.add(f)
            thr.setdefault(f, set()).add(float(s["border"]))
        for lv in t.get("leaf_values", []):
            leaves.add(float(np.float32(lv)))

    n_feat, n_leaf_vals = len(feats), len(leaves)
    max_thr = max((len(v) for v in thr.values()), default=1)
    mapping = n_feat * (4 + bits_width(n_input_features - 1) + bits_width(max_thr))
    thr_table = sum(len(thr[f]) * (1 if threshold_width(thr[f])[0] == 1 else 32)
                    for f in feats)
    leaf_table = n_leaf_vals * leaf_value_bits + (64 if leaf_value_bits < 32 else 0)

    leaf_ref, feat_ref = bits_width(n_leaf_vals - 1), bits_width(n_feat - 1)
    tree_bits = 0
    for d in depths:
        if oblivious:
            tree_bits += d * (feat_ref + bits_width(max_thr)) + (2 ** d) * leaf_ref
        else:
            tree_bits += leaf_ref * (2 ** d) + feat_ref * (2 ** d - 1)
            tree_bits += d * bits_width(max_thr)
    return mapping + thr_table + leaf_table + tree_bits


def run_catboost_grid(datasets, seeds, out_dir="results", ckpt_dir="checkpoints",
                      verbose=True) -> pd.DataFrame:
    import json
    from catboost import CatBoostClassifier, CatBoostRegressor
    from sklearn.metrics import accuracy_score, r2_score, brier_score_loss
    from .metrics import adaptive_ece

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    parts = []

    for ds, (task, X, y) in datasets.items():
        for seed in seeds:
            tag = f"cb_{ds}_s{seed}"
            if os.path.exists(_ckpt(ckpt_dir, tag)):
                parts.append(pd.read_csv(_ckpt(ckpt_dir, tag)))
                continue

            rows = []
            Xtr, Xte, ytr, yte = split(task, X, y, seed)
            CB = CatBoostClassifier if task == "clf" else CatBoostRegressor
            n_feat = Xtr.shape[1]

            for depth in CB_DEPTH:
                for n_iter in CB_ITERS:
                    m = CB(depth=depth, iterations=n_iter, verbose=0,
                           random_seed=seed, allow_writing_files=False).fit(Xtr, ytr)
                    tmp = os.path.join(ckpt_dir, "_cb_tmp.json")
                    m.save_model(tmp, format="json")
                    cbj = json.load(open(tmp))

                    if task == "clf":
                        p = m.predict_proba(Xte)[:, 1]
                        sc = dict(metric=accuracy_score(yte, (p > 0.5).astype(int)),
                                  ece=adaptive_ece(yte, p),
                                  brier=brier_score_loss(yte, p))
                    else:
                        sc = dict(metric=r2_score(yte, m.predict(Xte)),
                                  ece=np.nan, brier=np.nan)

                    n_nodes = sum(2 ** (len(t.get("splits", [])) + 1) - 1
                                  for t in cbj.get("oblivious_trees", []))
                    for arm, obl in [("catboost", False), ("catboost_oblivious", True)]:
                        rows.append(dict(
                            dataset=ds, seed=seed, arm=arm, depth=depth,
                            config=f"d{depth}_i{n_iter}",
                            reference_kb=kb(catboost_bits(cbj, n_feat, obl)),
                            succinct_kb=kb(catboost_bits(cbj, n_feat, obl)),
                            padding=1.0,
                            fp32_kb=kb(n_nodes * 128), fp16_kb=kb(n_nodes * 64), **sc))

            part = pd.DataFrame(rows)
            part.to_csv(_ckpt(ckpt_dir, tag), index=False)
            parts.append(part)
            if verbose:
                print(f"  {ds} seed={seed}: CatBoost done")

    grid = pd.concat(parts, ignore_index=True)
    grid.to_csv(os.path.join(out_dir, "grid_catboost.csv"), index=False)
    return grid
