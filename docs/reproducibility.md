# Reproducibility guide

## Environment

The lockfile is the authoritative dependency snapshot. From the repository root:

```powershell
uv sync --locked --extra dev --extra notebook --python 3.13
uv run python -c "import torch, torch_geometric, flowsheet_rl; print(torch.__version__, torch_geometric.__version__)"
uv run --extra dev pytest -q
```

The package supports Python 3.11–3.13. The supplied lockfile selects CPU PyTorch, so a GPU is not required.

## Deterministic inputs

Configuration lives in `configs/*.yaml`. Commands accept an explicit training seed and evaluation seed list. Experiment metadata records the effective configuration, source hashes, platform, and installed package versions. Evaluation uses fresh environments for fixed cases.

## Recreate the public example scale

```powershell
uv run python scripts/train.py --config configs/train_long.yaml --seed 7 --seeds 101 102 103 104 105 --out runs/long_seed_7
uv run python scripts/plot_training_results.py --run runs/long_seed_7
```

Training is stochastic. Identical software and seeds are designed to reproduce behavior on the same platform, but floating-point kernels and scatter operations can differ across operating systems, processors, or accelerator libraries. Compare configurations, summaries, and qualitative trends rather than assuming bitwise identity across machines.

## Evidence boundaries

- `examples/long_run` is a compact record, not a statistically powered benchmark.
- Notebook mini-runs illustrate mechanics; they are not additional independent replicates.
- The rule policy is a constructive baseline, not a learned target.
- The target simulator is a synthetic distribution shift, not a calibrated high-fidelity model.
