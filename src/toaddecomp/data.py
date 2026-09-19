"""Benchmark loading, with an on-disk cache so nothing is downloaded twice."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.datasets import (load_breast_cancer, fetch_california_housing,
                              fetch_covtype, fetch_openml)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

__all__ = ["LOADERS", "load_all", "split"]

MAX_TRAIN_ROWS = 30_000
MAX_TEST_ROWS = 20_000


def _openml_numeric(name: str, version: int = 1):
    d = fetch_openml(name, version=version, as_frame=True)
    X = d.data.copy()
    cat = X.select_dtypes(include=["category", "object"]).columns
    if len(cat):
        X[cat] = OrdinalEncoder().fit_transform(X[cat].astype(str))
    return X.fillna(-1).to_numpy(float), np.asarray(pd.Categorical(d.target).codes)


LOADERS = {
    "breast_cancer":      ("clf", lambda: load_breast_cancer(return_X_y=True)),
    "california_housing": ("reg", lambda: (lambda d: (d.data, d.target))(fetch_california_housing())),
    "covertype_binary":   ("clf", lambda: (lambda d: (d.data, (d.target == 2).astype(int)))(fetch_covtype())),
    "mushroom":           ("clf", lambda: _openml_numeric("mushroom")),
    "kr-vs-kp":           ("clf", lambda: _openml_numeric("kr-vs-kp")),
}


def load_all(cache_dir: str = "cache", verbose: bool = True):
    """Load every benchmark, caching each as an .npz after the first fetch."""
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for name, (task, fn) in LOADERS.items():
        path = os.path.join(cache_dir, f"{name}.npz")
        try:
            if os.path.exists(path):
                z = np.load(path, allow_pickle=True)
                X, y = z["X"], z["y"]
                src = "cache"
            else:
                X, y = fn()
                X, y = np.asarray(X, float), np.asarray(y)
                np.savez_compressed(path, X=X, y=y)
                src = "downloaded"
            out[name] = (task, X, y)
            if verbose:
                print(f"  {name:20s} {X.shape}  {task}  ({src})")
        except Exception as exc:
            print(f"  {name:20s} skipped ({type(exc).__name__}: {exc})")
    return out


def split(task, X, y, seed: int):
    """Stratified for classification, capped so the grid stays tractable."""
    strat = y if task == "clf" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=seed, stratify=strat)
    rng = np.random.RandomState(seed)
    if len(Xtr) > MAX_TRAIN_ROWS:
        i = rng.choice(len(Xtr), MAX_TRAIN_ROWS, replace=False)
        Xtr, ytr = Xtr[i], ytr[i]
    if len(Xte) > MAX_TEST_ROWS:
        i = rng.choice(len(Xte), MAX_TEST_ROWS, replace=False)
        Xte, yte = Xte[i], yte[i]
    return np.asarray(Xtr, float), np.asarray(Xte, float), ytr, yte
