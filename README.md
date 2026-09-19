# A Four-Factor Decomposition of Memory Compression in Boosted Tree Ensembles

Replication package for the paper of the same name.

Published compression figures for compact tree ensembles bundle several
separate choices into a single number, so it is hard to tell what actually
delivers the saving. This work separates four factors and measures each on its
own: the learner, the model size settings, the storage layout, and the coding
of split thresholds and leaf values.

Every ensemble is priced under a bit-exact reimplementation of the layout
published by Herrmann et al. (ICLR 2026), so arms that never touch the training
algorithm can be compared against arms that do, without forking LightGBM.

## What the Study Finds

The storage layout on its own returns 3.7x to 4.8x against a float32 baseline,
with unmodified training. Clustering leaf values lifts that to as much as
19.7x. Clustering split thresholds returns close to nothing, and the size of
that nothing tracks feature cardinality. A succinct (LOUDS) tree encoding
removes the padding that fixed-depth array layouts pay for, but only past a
measured depth: it costs space up to depth four and reaches 2.48x at depth
eight. Eight-bit leaf quantisation is free for classification and breaks for
regression.

`docs/RESULTS.md` has the full tables. `docs/ASSUMPTIONS.md` separates what was
verified against the reference source from what remains a modelling choice.

## Install

Python 3.9 or newer.

```bash
git clone <repository-url>
cd toad-decomposition
pip install -r requirements.txt
```

## Reproduce

```bash
python -m pytest tests/ -q        # 5 tests, under a minute
python scripts/run_all.py         # full run, 12 seeds, about 90 minutes
```

For a quick check first:

```bash
python scripts/run_all.py --seeds 1    # about 8 minutes
```

Results land in `results/`, figures in `figures/`. Benchmarks download once and
cache in `cache/`.

The run checkpoints after every dataset and seed pair, so an interrupted run
resumes instead of restarting. Delete a file from `checkpoints/` to force that
cell to recompute. Checkpoints are keyed on dataset and seed rather than on
grid contents, so **clear `checkpoints/` entirely after editing the grid** in
`src/toaddecomp/experiment.py`. Skipping this silently reuses stale results.

The first thing to run after any change to the traversal or coding code is
`tests/test_predictor.py`. It checks that the vectorised predictor reproduces
LightGBM's own output exactly. If that fails, every quality number for a
post-hoc edited model is wrong.

## Layout

```
src/toaddecomp/
  trees.py        parsing, traversal, vectorised prediction
  accounting.py   reference and succinct layouts, in bits
  coding.py       threshold clustering, leaf clustering, leaf quantisation
  data.py         benchmark loading with an on-disk cache
  metrics.py      accuracy, R-squared, adaptive ECE
  experiment.py   the grid, checkpointed
  analysis.py     iso-quality ratios, depth table, sensitivity
  figures.py      every figure, sized for one IEEE column
scripts/run_all.py
tests/
figures/          figures as they appear in the paper
docs/             assumptions and full results
notebooks/        Colab version of the same pipeline
```

## Benchmarks

Breast Cancer, California Housing and Covertype ship with scikit-learn.
Mushroom and KR-vs-KP come from OpenML. All five are public and download
automatically. Covertype is subsampled to 30,000 training rows to keep the grid
tractable, which is a deliberate scope decision and is stated in the paper.

## Notes on Comparison

Compression is read horizontally. For a fixed quality level, the smallest model
each arm needs is found, and the ratio is taken against **that learner's own**
float32 baseline. Pricing CatBoost against a LightGBM baseline would fold a
difference in learner quality into a figure labelled compression, so the
cross-learner comparison is reported separately and labelled as model
selection.

Memory is counted, not measured on device. Latency and energy are not measured
at all. The succinct layout adds a rank operation per level that would be paid
at inference time, and that cost is unmeasured here.

## Citation

See `CITATION.cff`. The paper reference will be added once the proceedings
are published.

## License

MIT. See `LICENSE`.
