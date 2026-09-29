# Dependency Notes

Do not freeze versions until implementation starts. At that point, verify current compatibility and create a lock file.

Initial candidates:

- Python 3.11 or another version supported by the selected scientific stack;
- NumPy;
- NetworkX;
- Gymnasium;
- Stable-Baselines3 and SB3 Contrib for `MaskablePPO`;
- PyTorch through the selected RL stack;
- Pydantic or standard-library dataclasses;
- pandas or Polars;
- pytest;
- Ruff and a static type checker.

Official implementation notes:

- Gymnasium custom environments must define observation and action spaces and follow the `reset`/`step` API.
- `MaskablePPO` requires mask-aware evaluation utilities. When using subprocess vectorization, implement `action_masks` in the environment rather than relying on an external wrapper.

Official references checked on 2026-09-20:

- Gymnasium custom environment guide: <https://gymnasium.farama.org/introduction/create_custom_env/>
- Gymnasium action-masking tutorial: <https://gymnasium.farama.org/tutorials/training_agents/action_masking_taxi/>
- SB3 Contrib MaskablePPO documentation: <https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html>

Record package versions, platform, and installation command with every experiment series.

## Implemented stack (2026-09-21)
The active dependency specification is pyproject.toml plus uv.lock. The project uses Gymnasium, NetworkX, CPU PyTorch and PyTorch Geometric with a transparent joint-likelihood PPO implementation. SB3 Contrib was not used because the requested conditional hybrid graph policy requires explicit likelihood and entropy handling. Exact tested versions are saved with each experiment and in ../07_RESULTS/20260921_foundation_validation.md. The initial candidate list above is retained as history.
