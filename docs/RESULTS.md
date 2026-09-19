# Headline Numbers

Produced by `python scripts/run_all.py` with the default twelve seeds, from
5,760 fitted LightGBM ensembles and 32,640 scored configurations plus the
CatBoost grid. Figures move slightly with library versions; the ordering does
not.

## Compression at Matched Quality

Each learner priced against its own float32 baseline.

| Technique | Breast | Calif. | Cover | KR-KP | Mush. |
|---|---|---|---|---|---|
| 16-bit baseline | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 |
| Reference layout | 3.75 | 4.03 | 3.72 | 3.83 | 4.84 |
| Succinct layout | 3.65 | 3.72 | 3.86 | 5.27 | 4.98 |
| Layout + leaf clustering | 9.41 | 7.73 | 8.40 | 13.38 | 19.73 |
| Succinct + leaf clustering | 7.50 | 6.56 | 8.11 | 14.38 | 13.95 |
| Succinct + 8-bit leaves | 7.25 | 6.87 | 7.83 | 13.02 | 12.67 |

The storage layout alone returns 3.72x to 4.84x with unmodified training.

## Succinct Layout by Depth

| Depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Padding | 1.00 | 1.00 | 1.10 | 1.30 | 1.70 | 2.50 | 4.01 | 6.90 |
| Size reduction | 0.92 | 0.92 | 0.94 | 0.99 | 1.08 | 1.29 | 1.69 | 2.48 |
| Loses (%) | 100 | 100 | 100 | 61 | 36 | 27 | 14 | 0 |

The crossover sits between depth four and depth five. At depth eight the
succinct layout does not lose on any of the 4,080 models at that setting.

## Value Coding

Leaf clustering lifts compression to between 7.73x and 19.73x. Threshold
clustering returns close to nothing, and the size of that nothing tracks
feature cardinality: on the two ordinal-encoded categorical datasets it changed
stored size by 0.1 percent or less, because most features already carried fewer
distinct thresholds than the cluster count.

Paired tests across seeds at a 4 KB budget put leaf clustering ahead of the
plain layout on four of five benchmarks. Threshold clustering is never
significantly ahead, and is significantly behind on California Housing.

## Leaf Precision

Matched against the same base configuration.

| Width | Mean quality change | Worst case | Note |
|---|---|---|---|
| 16-bit | 0.0000 | 0.0000 | exactly lossless everywhere |
| 8-bit | -0.0008 to +0.0002 | -0.0104 | safe on all five |
| 4-bit | -0.0006 (clf) / -5.57 (reg) | -27.5 | fails on regression |

Regression leaf values span a far wider range than log-odds, so a uniform scale
runs out of resolution. A per-tree scale would probably fix it.

## Cross-Learner

At matched quality the best CatBoost model is 1.44x to 3.52x smaller than the
best LightGBM model. This is a model selection result, not a compression one,
and it is reported separately for that reason. Oblivious trees are complete by
construction, so they pay no padding and gain nothing from the succinct
encoding.
