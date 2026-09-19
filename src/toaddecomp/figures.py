"""Every figure in the paper, sized for a single IEEE column."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import BUDGETS, envelope

__all__ = ["apply_style", "envelope_figure", "coding_figure", "depth_figure", "make_all"]

COL_W, COL_H = 3.45, 2.65

PRETTY = {"breast_cancer": "Breast Cancer", "california_housing": "California Housing",
          "covertype_binary": "Covertype (binary)", "mushroom": "Mushroom",
          "kr-vs-kp": "KR-vs-KP"}

CURVES = [
    ("LightGBM fp32",       "lgb",                "fp32_kb",      "o", "#999999", ":"),
    ("LightGBM fp16",       "lgb",                "fp16_kb",      "v", "#666666", ":"),
    ("Reference layout",    "lgb",                "reference_kb", "s", "#2C5F8D", "-"),
    ("Succinct layout",     "lgb",                "succinct_kb",  "^", "#3F7A5E", "-"),
    ("+ leaf clustering",   "lgb+leaf",           "reference_kb", "D", "#B5533C", "-"),
    ("CatBoost oblivious",  "catboost_oblivious", "reference_kb", "*", "#7B5EA7", "--"),
]

CODING_ARMS = [("no value coding",     "lgb",       "o", "#4A4A4A"),
               ("leaf clustering",     "lgb+leaf",  "^", "#B5533C"),
               ("threshold clustering", "lgb+thr",  "s", "#2C5F8D"),
               ("leaf quantization",   "lgb+quant", "D", "#3F7A5E")]


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.labelsize": 7.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "legend.fontsize": 6.2, "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "grid.linewidth": 0.5, "lines.linewidth": 1.2,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=400)
    plt.close(fig)


def envelope_figure(grid, dataset, task, out_dir="figures"):
    d = grid[grid.dataset == dataset]
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    for label, arm, column, marker, colour, ls in CURVES:
        sub = d[d.arm == arm]
        if not len(sub) or column not in sub.columns:
            continue
        y = envelope(sub, column)
        ok = np.isfinite(y)
        if ok.sum() < 2:
            continue
        ax.plot(BUDGETS[ok], y[ok], marker=marker, color=colour, linestyle=ls,
                markersize=3.4, markeredgewidth=0, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("memory budget (KB)")
    ax.set_ylabel("accuracy" if task == "clf" else r"$R^2$")
    ax.set_title(PRETTY.get(dataset, dataset))
    ax.grid(alpha=0.28, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right", handlelength=1.8,
              borderpad=0.2, labelspacing=0.28)
    _save(fig, out_dir, f"fig_envelope_{dataset}")


def coding_figure(grid, dataset, task, out_dir="figures"):
    d = grid[grid.dataset == dataset]
    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    floor = None
    for label, arm, marker, colour in CODING_ARMS:
        sub = d[d.arm == arm]
        if not len(sub):
            continue
        g = sub.groupby("config", as_index=False).agg({"reference_kb": "mean", "metric": "mean"})
        ax.scatter(g.reference_kb, g.metric, s=7, marker=marker, color=colour,
                   alpha=0.6, linewidths=0, label=label)
        floor = g.metric.quantile(0.02)
    ax.set_xscale("log")
    ax.set_xlabel("size under reference layout (KB)")
    ax.set_ylabel("accuracy" if task == "clf" else r"$R^2$")
    ax.set_title(PRETTY.get(dataset, dataset))
    ax.grid(alpha=0.28, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right", handlelength=1.2,
              borderpad=0.2, labelspacing=0.28, scatterpoints=1)
    if floor is not None and np.isfinite(floor):
        ax.set_ylim(bottom=floor)
    _save(fig, out_dir, f"fig_coding_{dataset}")


def depth_figure(grid, out_dir="figures"):
    """The headline: where the succinct layout starts to pay."""
    d = grid[grid.arm == "lgb"].copy()
    d["gain"] = d.reference_kb / d.succinct_kb
    g = (d.groupby("depth")
          .agg(gain=("gain", "mean"), worse=("gain", lambda s: (s < 1).mean()))
          .reset_index())

    fig, ax = plt.subplots(figsize=(COL_W, COL_H))
    ax.plot(g.depth, g.gain, marker="o", color="#2C5F8D", markersize=4,
            markeredgewidth=0, label="size reduction")
    ax.axhline(1.0, color="#B5533C", linewidth=0.8, linestyle="--")
    ax.text(g.depth.max() - 0.1, 1.0, "no benefit", fontsize=6.2,
            color="#B5533C", va="bottom", ha="right")
    ax.set_xlabel("configured max_depth")
    ax.set_ylabel("size reduction from succinct layout")
    ax.set_xticks(g.depth.astype(int).tolist())
    ax.grid(alpha=0.28, linewidth=0.5)

    ax2 = ax.twinx()
    ax2.plot(g.depth, g.worse * 100, marker="s", color="#999999", markersize=3.2,
             linestyle=":", markeredgewidth=0, label="models where it loses")
    ax2.set_ylabel("% of models where it loses", fontsize=7.0, color="#666666")
    ax2.tick_params(axis="y", labelsize=6.8, colors="#666666")
    ax2.set_ylim(-4, 104)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left",
              handlelength=1.8, borderpad=0.2, labelspacing=0.28)
    _save(fig, out_dir, "fig_depth_gain")


def make_all(grid, datasets, out_dir="figures"):
    apply_style()
    for ds in grid.dataset.unique():
        task = datasets[ds][0] if ds in datasets else "clf"
        envelope_figure(grid, ds, task, out_dir)
        coding_figure(grid, ds, task, out_dir)
    depth_figure(grid, out_dir)
    print(f"figures written to {out_dir}/")
