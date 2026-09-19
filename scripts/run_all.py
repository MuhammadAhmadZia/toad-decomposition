#!/usr/bin/env python3
"""Reproduce every number and figure in the paper.

    python scripts/run_all.py                 # full run, 12 seeds, ~90 min
    python scripts/run_all.py --seeds 1       # smoke test, ~8 min
    python scripts/run_all.py --skip-catboost

Results land in results/, figures in figures/, checkpoints in checkpoints/.
An interrupted run resumes: finished cells load from checkpoints. Delete a
checkpoint file to force that cell to recompute. Change the grid in
src/toaddecomp/experiment.py and you must clear checkpoints/ as well, because
checkpoints are keyed on dataset and seed, not on grid contents.
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from toaddecomp import data, experiment, analysis, figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 13)))
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--skip-catboost", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("Loading benchmarks")
    datasets = data.load_all(args.cache)
    if not datasets:
        sys.exit("No datasets loaded. Check the network connection.")

    print(f"\nLightGBM grid over {len(args.seeds)} seeds")
    grid = experiment.run_lightgbm_grid(datasets, args.seeds, args.out, args.checkpoints)

    if not args.skip_catboost:
        print("\nCatBoost grid")
        cb = experiment.run_catboost_grid(datasets, args.seeds, args.out, args.checkpoints)
        grid = pd.concat([grid, cb], ignore_index=True)

    os.makedirs(args.out, exist_ok=True)
    grid.to_csv(os.path.join(args.out, "all_results.csv"), index=False)
    print(f"\n{len(grid)} scored configurations -> {args.out}/all_results.csv")

    print("\nTable II. Compression at matched quality against the same learner's fp32\n")
    comp = analysis.compression_table(grid)
    print(comp.round(2).to_string())
    comp.to_csv(os.path.join(args.out, "table2_compression.csv"))

    print("\n\nTable III. Succinct layout by configured depth\n")
    depth = analysis.depth_table(grid)
    print(depth.to_string())
    depth.to_csv(os.path.join(args.out, "table3_depth.csv"))

    print("\n\nCross-learner advantage (model selection, NOT compression)\n")
    cross = analysis.cross_learner_table(grid)
    print(cross.round(2).to_string() if len(cross) else "  (no overlapping quality range)")
    cross.to_csv(os.path.join(args.out, "cross_learner.csv"))

    print("\n\nSensitivity of the succinct-layout result to accounting choices\n")
    sens = analysis.sensitivity_table(datasets, args.seeds, experiment.STRUCT)
    print(sens.round(2).to_string())
    sens.to_csv(os.path.join(args.out, "sensitivity.csv"))

    print("\nBuilding figures")
    figures.make_all(grid, datasets, args.figures)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
