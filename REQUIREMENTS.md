# Project Requirements

This document describes the runtime, package, data, and execution requirements for the LA-HRL + ALNS hybrid FSMVRP evaluation utilities.

## 1. Runtime Requirements

### Python

- Python 3.10 or newer.
- A virtual environment is recommended.
- Commands should be run from the repository root.

### Operating System

- Windows is supported and appears to be the primary development environment.
- Linux and macOS should work when the local tensor paths are available.
- Multiprocessing is used through `concurrent.futures.ProcessPoolExecutor`; Windows users should run scripts from a normal terminal or IDE run configuration.

### Pip Packages

Required packages are listed in `requirements.txt`:

```text
numpy>=2.0
torch>=2.0
alns>=7.0
```

PyTorch CPU builds are sufficient for pure ALNS and CPU inference. For GPU inference, install a CUDA-enabled PyTorch build that matches the local CUDA runtime.

## 2. Model Checkpoint Requirements

The hybrid runner requires an LA-HRL checkpoint compatible with `FSMVRPTester_PPO`.

Default path:

```text
./results/train_20nodes
```

Bundled checkpoint path:

```text
baseline/checkpoint-best.pt
```

Recommended command for the bundled checkpoint:

```bash
python hybridtest.py --checkpoint_path baseline --checkpoint_epoch best
```

The tester implementation must map `checkpoint_epoch=best` to `checkpoint-best.pt`.

## 3. Data Requirements

### Bundled Test Tensors

The repository includes synthetic test tensors under `data/`:

```text
test_tensor(20)_6_100_5678.pt
test_tensor(50)_6_100_5678.pt
test_tensor(75)_6_100_5678.pt
test_tensor(100)_6_100_5678.pt
test_tensor(200)_6_100_5678.pt
test_tensor(500)_6_100_5678.pt
test_tensor(1000)_6_100_5678.pt
```

These files are enough to run the test tensor workflow.

## 4. Functional Requirements

### Hybrid Evaluation

The hybrid workflow must:

- load local tensor files with depot coordinates, node coordinates, node demand, vehicle capacities, fixed costs, and variable costs;
- run LA-HRL inference through `FSMVRPTester_PPO`;
- extract route-level assignments, vehicle types, delivered demand, and objective cost;
- convert LA-HRL routes to ALNS-compatible `FSMVRP_State` objects;
- validate vehicle type ranges and capacity violations before ALNS starts;
- run ALNS improvement with fixed seeds and configured destroy/repair operators;
- report LA-HRL, scratch ALNS, hybrid ALNS, and gap metrics where available;
- write logs and CSV exports under a timestamped `logs/` directory.

### Pure ALNS Ablation

The pure ALNS workflow must:

- load the same tensor structure;
- build initial ALNS states without LA-HRL initialization;
- run the ALNS operator set with fixed seeds;
- report pure ALNS objective values and gap metrics;
- write logs, route exports, and operator statistics.

### Bridge Conversion

The bridge utility must:

- accept LA-HRL route dictionaries;
- preserve 1-indexed customer IDs expected by ALNS;
- add depot padding at the start and end of every route;
- detect invalid vehicle type indexes;
- detect route loads exceeding vehicle capacity beyond a configurable tolerance;
- mark missing customers as unassigned;
- verify that the resulting ALNS state has a finite objective value.

## 5. Output Requirements

Runs must create timestamped output directories:

```text
logs/run_<timestamp>_<dataset>_<method>/
```

Expected output files include:

- console transcript;
- structured run log;
- `operator_stats.csv`;
- `best_routes.csv`.

## 6. Reproducibility Requirements

- ALNS seed lists are fixed in the scripts.
- Multi-run summaries should report mean, standard deviation, best run, and gap metrics where available.
- `.pt` files should be treated as trusted inputs because they are loaded with `torch.load`.
- The exact PyTorch, NumPy, and ALNS versions should be recorded when reporting final experiment results.

## 7. Verification Checklist

Before running a full evaluation:

1. Create and activate a Python 3.10+ virtual environment.
2. Install `requirements.txt`.
3. Confirm the checkpoint path exists.
4. Run a small test tensor case first.
5. Inspect the generated `logs/` directory.
