# LA-HRL + ALNS Hybrid FSMVRP Evaluation

This repository contains evaluation utilities for the Fleet Size and Mix Vehicle Routing Problem (FSMVRP). It provides:

- a hybrid runner that converts LA-HRL model solutions into ALNS states and improves them with Adaptive Large Neighborhood Search;
- a pure ALNS ablation runner for comparing against the hybrid workflow;
- a bridge utility that validates LA-HRL routes before passing them into the ALNS cost model;
- bundled test tensor datasets for local evaluation.

## Repository Layout

```text
.
|-- ablation_test.py                  # Pure ALNS evaluation and ablation runner
|-- bridge.py                         # LA-HRL solution -> ALNS FSMVRP_State adapter
|-- hybridtest.py                     # Sequential LA-HRL inference -> ALNS improvement runner
|-- __init__.py                       # Package export for the bridge helper
|-- baseline/
|   `-- checkpoint-best.pt            # Bundled LA-HRL checkpoint artifact
|-- data/
|   `-- test_tensor(...).pt           # Synthetic test tensor files
`-- requirements.txt                  # Runtime package requirements
```

The bundled test tensor files can be used directly.

## What The Scripts Do

### `hybridtest.py`

Runs a sequential LA-HRL -> ALNS evaluation:

1. Loads a local test tensor file.
2. Runs LA-HRL inference through `FSMVRPTester_PPO`.
3. Captures the route solution produced by the tester.
4. Converts the LA-HRL route representation to an ALNS `FSMVRP_State` using `bridge.py`.
5. Runs ALNS with multiple seeds and custom destroy/repair operators.
6. Writes detailed logs, operator statistics, and route exports.

### `ablation_test.py`

Runs a pure ALNS baseline using the same menu, logging format, operators, seed handling, and output structure. This is useful for measuring how much the LA-HRL initialization contributes before ALNS improvement.

### `bridge.py`

Validates and converts LA-HRL extracted routes into the ALNS state format:

- checks vehicle type bounds;
- checks route load against vehicle capacity;
- depot-pads each route as `[0, ...customers, 0]`;
- tracks unassigned customers;
- calls `FSMVRP_State.objective()` once to verify the constructed state is usable.

## Data

The repository is documented for the bundled local test tensors in `data/`. No external dataset is required for the documented workflow.

## Installation

Python 3.10 or newer is recommended because the code uses modern type hint syntax.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA evaluation, install the PyTorch wheel that matches your CUDA runtime from the official PyTorch installation selector, then keep the rest of the packages from `requirements.txt`.

## Checkpoint Setup

`hybridtest.py` defaults to:

```text
./results/train_20nodes
```

This repository includes:

```text
baseline/checkpoint-best.pt
```

If you want to use the bundled checkpoint directly, run the hybrid script with:

```bash
python hybridtest.py --checkpoint_path baseline --checkpoint_epoch best
```

Your tester implementation must resolve `checkpoint_epoch=best` to `checkpoint-best.pt`, which is the naming convention used by the bundled artifact.

## Running The Hybrid Evaluation

```bash
python hybridtest.py --checkpoint_path baseline --checkpoint_epoch best
```

Useful options:

```bash
python hybridtest.py --cuda
python hybridtest.py --log_level DEBUG
python hybridtest.py --no_reuse_pool
```

The script opens an interactive menu where you select:

- bundled test tensor size;
- number of instances to test from each file;
- number of repeated runs.

## Running The Pure ALNS Ablation

```bash
python ablation_test.py
```

Useful options:

```bash
python ablation_test.py --log_level DEBUG
python ablation_test.py --no_reuse_pool
```

The menu mirrors the hybrid runner but skips LA-HRL inference and starts ALNS from a scratch initial state.

## Output Files

Each run creates a timestamped directory under:

```text
logs/run_<timestamp>_<dataset>_<method>/
```

Typical outputs include:

- `console_output_<timestamp>.txt` - full console transcript;
- `hybrid_run_<timestamp>.log` or `pure_alns_run_<timestamp>.log` - structured logger output;
- `operator_stats.csv` - ALNS destroy/repair operator weights and usage statistics;
- `best_routes.csv` - best route details per instance.

## Bundled Data

The `data/` directory includes test tensor files for the following sizes:

- 20 nodes
- 50 nodes
- 75 nodes
- 100 nodes
- 200 nodes
- 500 nodes
- 1000 nodes

The scripts use internal baseline maps for these tensors and report gaps against those values in the run logs.

## Notes For Reproducibility

- ALNS runs use multiple fixed seeds.
- Worker pools are reused by default to reduce Windows multiprocessing overhead.
- Pass `--no_reuse_pool` when debugging worker hangs or memory growth.
- The scripts use `torch.load(..., weights_only=False)` for tensor files, so only load trusted `.pt` files.
- Run commands from this repository root.

## Troubleshooting

### CUDA is not used

`--cuda` only enables GPU inference when `torch.cuda.is_available()` returns `True`. Install a CUDA-enabled PyTorch build if needed.

### Multiprocessing issues on Windows

Run from a normal terminal with the virtual environment activated. If a worker process hangs while debugging, retry with:

```bash
python hybridtest.py --no_reuse_pool
python ablation_test.py --no_reuse_pool
```
