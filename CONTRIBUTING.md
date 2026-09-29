# Contributing

Thank you for helping improve this educational project. Contributions should keep the physical rules, learned decisions, and optimization logic visibly separate.

## Development setup

```powershell
uv sync --locked --extra dev --extra notebook --python 3.13
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```

## Change expectations

- Add or update tests when changing state, actions, masks, simulation equations, reward terms, termination, checkpointing, or evaluation.
- Record assumptions and units near each process equation.
- Use fixed seeds and save configuration and package versions for experiments.
- Treat notebook examples as interfaces to importable code; keep core model logic in `src/flowsheet_rl`.
- Label synthetic values and preliminary results clearly.
- Link primary literature rather than committing publisher PDFs.

Open an issue before a large feature such as recycle handling, rigorous thermodynamics, external simulator coupling, a new action hierarchy, or a different RL algorithm. A pull request should explain the behavior change, scientific assumptions, and validation performed.
