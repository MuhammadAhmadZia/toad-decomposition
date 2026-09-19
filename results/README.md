# Results

This directory is populated by `python scripts/run_all.py`.

| File | Contents |
|---|---|
| `all_results.csv` | every scored configuration, one row each |
| `grid_lightgbm.csv` | LightGBM arms only |
| `grid_catboost.csv` | CatBoost arms only |
| `table2_compression.csv` | compression at matched quality, Table II |
| `table3_depth.csv` | succinct layout by depth, Table III |
| `cross_learner.csv` | CatBoost advantage, model selection not compression |
| `sensitivity.csv` | the depth result under varied accounting assumptions |

Columns in `all_results.csv`:

`dataset`, `seed`, `arm`, `depth`, `config`, `reference_kb`, `succinct_kb`,
`padding`, `fp32_kb`, `fp16_kb`, `metric`, `ece`, `brier`.

`metric` is accuracy for classification and R-squared for regression. `padding`
is slots charged per real node under the configured-depth accounting.

The CSVs from the run behind the paper are archived with the Zenodo release
rather than committed here, since the full grid is large.
