# AlphaHRL

The codebase focuses on hierarchical reinforcement learning for quantitative factor discovery.

## Contents

- Low-level policy for generating and selecting symbolic alpha factors
- High-level policy for allocating over a discovered factor pool
- Data collection and Qlib-format preprocessing utilities
- Qlib-based evaluation scripts

## Repository Layout

```text
scripts/
  low_level.py         main low-level training entry
  high_level.py        main high-level training entry

AlphaHRL/
  HRL/                 core hierarchical RL implementation
  alphahrl_qlib/       Qlib data interface used by the main scripts

data_collection/
  fetch_baostock_data.py   public-data download and preprocessing
  qlib_dump_bin.py         CSV-to-Qlib conversion utility

out/                   example logs, checkpoints, and result files
data/                  cached or regenerated data files
```

## Setup

We recommend using a clean Python environment. 

```bash
pip install -r requirements.txt
```

If your platform has trouble building some packages, a Conda environment is also acceptable. The repository depends mainly on PyTorch, Qlib, Stable-Baselines3, `sb3-contrib`, and the packages listed in `requirements.txt`.

## Data Preparation

The main experiments use Qlib-format daily China A-share data.

### Option 1: Use an existing Qlib data directory

If you already have a local Qlib dataset, point the code to that directory.

Important: the current code contains a few hard-coded local paths from the development environment. Before running experiments, search the repository for:

```text
cn_data_rolling
```

and replace those occurrences with your local Qlib data path.

This affects at least:

- `scripts/low_level.py`
- `scripts/high_level.py`
- `AlphaHRL/alphahrl_qlib/stock_data.py`


### Option 2: Rebuild the data from public sources

The repository includes scripts for downloading market data from BaoStock and converting it to Qlib format:

```bash
python data_collection/fetch_baostock_data.py
```

Before running this script, set the output directories near the bottom of `data_collection/fetch_baostock_data.py` so they match your machine.

## Main Reproduction Commands

### 1. Low-Level Training

This stage learns a factor pool through symbolic-expression generation and selection.

Minimal example:

```bash
python scripts/low_level.py 
```

Results are written to:

```text
out/results/low_level/<run_name>/
out/tensorboard/low_level/
```

Key output files include checkpointed pool snapshots such as `*_steps_pool.json` and the selected `best_pool.json`.

### 2. High-Level Training

This stage learns portfolio allocation on top of a low-level factor pool.

Recommended usage is to explicitly pass the low-level result directory:

```bash
python scripts/high_level.py
```

Results are written to:

```text
out/results/high_level/<run_name>/
```

If you already have a fixed factor pool JSON file, you can provide `--pool_path=<path>` instead of `--low_level_result_dir`.

## Optional Modules

### AlphaGen

`AlphaGen` here refers to the earlier factor-generation baseline implemented in `alphagen_scripts/rl.py`.

Before running:

- replace the hard-coded Qlib path in `alphagen_scripts/rl.py`
- make sure a CUDA device is available, or manually change the device string in the script

Reproduction command:

```bash
python alphagen_scripts/rl.py
```


### DSO

The DSO baseline is implemented in `dso/dso.py`.


You will also need the project-side symbolic-expression and Qlib dependencies available in that environment.

Before running:

- ensure the Qlib default path used by `alphagen_qlib/stock_data.py` is valid on your machine
- adjust the CUDA device in `dso/dso.py` if needed

Reproduction command:

```bash
python dso/dso.py 
```

The script evaluates symbolic expressions online and prints the test IC every 100 evaluations. Its final output is the dictionary `ev.results`, which maps evaluation count to test performance. 

### GPlearn

The GPlearn baseline is implemented in `gplearn/gp.py`.

Reproduction command:

```bash
python gplearn/gp.py
```

This baseline uses `gplearn.genetic.SymbolicRegressor` with a custom IC-based fitness and periodically exports the best factor pool.

Outputs are written to:

```text
out/gp/<seed>/
```

### Other baselines

The following methods can be reproduced by referring to the author's code repository.
1.[AlphaSAGE](https://github.com/BerkinChen/AlphaSAGE)
2.[AlphaQCM](https://github.com/ZhuZhouFan/AlphaQCM)
3.[AlphaAgent](https://github.com/RndmVariableQ/AlphaAgent)


## Existing Artifacts

The repository may already contain logs, figures, intermediate checkpoints, or cached outputs under `out/` and `data/`. These files are included only as examples or cached artifacts. They are not required for understanding the code structure, and experiments can be rerun from scratch following the steps above.
