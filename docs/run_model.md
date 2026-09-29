# Run, explore, and train the flowsheet model

The environment is ready to use. The saved neural policy is an early smoke-test checkpoint: it does not yet generate a feasible product design in the five recorded evaluation cases. Start with the notebook and rule-based design to understand the process, then run a measured training pilot. A longer run is an experiment, not a guarantee of convergence.

All commands below are PowerShell commands run from the repository root. Outputs must use a **new directory** on each invocation; the `runs/` directory is ignored by Git.

## 1. Open the project and install

```powershell
Set-Location 'path\to\flowsheet-rl-educational'
uv sync --locked --extra dev --extra notebook --python 3.13
```

The local `.venv` is already installed. Python 3.13.11 and CPU PyTorch were used for validation. No commercial simulator or GPU is needed. `uv.lock` pins the environment, including the optional notebook dependencies. Once installed, the commands below call this environment directly; activating it is optional.

After `uv sync`, restart any notebook kernel and any VS Code Python test process that was already running. The sync can replace NumPy files on disk; an old process that keeps part of the previous NumPy version in memory can then fail with misleading errors such as `module 'numpy' has no attribute 'matrix'`. In VS Code, use **Developer: Reload Window**, then select `.venv/Scripts/python.exe` again. This check must print a NumPy version and `[1]` before testing:

```powershell
.\.venv\Scripts\python.exe -c "import numpy as np; print('NumPy:', np.__version__); print('flatnonzero:', np.flatnonzero(np.array([False, True])))"
```

On Windows/OneDrive, keep test scratch files inside a fresh project-local runtime directory. This avoids stale permissions in `%TEMP%` and `.pytest_cache`:

```powershell
$testRun = ".runtime\pytest-$([guid]::NewGuid().ToString('N'))"
$env:MPLCONFIGDIR = Join-Path (Get-Location) ".runtime\matplotlib"
New-Item -ItemType Directory -Force -Path $testRun,$env:MPLCONFIGDIR | Out-Null
.\.venv\Scripts\python.exe -m pytest -q --basetemp $testRun -p no:cacheprovider
```

The Conda warning `Ignoring SSL_CERT_DIR` is harmless when `uv sync` finishes with `Resolved` and `Checked`. If a future download actually fails, remove the stale variable for that terminal with `Remove-Item Env:SSL_CERT_DIR -ErrorAction SilentlyContinue` and rerun `uv sync`.

## 2. Interact through the notebook

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks/learn_flowsheet_rl_step_by_step.ipynb
```

For a slower teaching path that reveals the source functions, custom equations, graph tensors, GNN decisions, a 768-transition learning demonstration, and the saved long-run evaluation, open:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks/learn_flowsheet_rl_step_by_step.ipynb
```

Select the project's Python environment if your editor asks for a kernel. Run the cells from top to bottom. The notebook lets you:

- Adjust feed amounts, variability, temperature, fidelity and random seed.
- Generate the rule-based design, inspect its flowsheet and stream table, and see each reward contribution.
- Build a design one action at a time, inspecting valid choices before selecting a unit and its normalized parameter.
- Inspect actual graph tensors, neural probabilities, critic values and a message-passing calculation.
- Load the saved checkpoint and compare it with rule and random policies on the same feed cases.
- Optionally train for 64 transitions by changing `RUN_TRAINING = False` to `True`.

Each session creates a timestamped folder under `runs/notebook_sessions`. The manual cells operate on an independent environment: reset that environment before starting another manual design. A thermal parameter of `90/170` means 370 K; reactor and separator parameters have different meanings despite sharing the interval [0,1]. The displayed masks tell you which units are currently possible.

The committed teaching notebook retains representative outputs from an end-to-end validation, including the opt-in training cell. Those outputs are diagnostic examples, not additional independent evidence of learned performance.

For a reproducible noninteractive execution, use a new output filename:

```powershell
.\.venv\Scripts\python.exe scripts/execute_notebook.py --out runs/notebook_validation/my_executed.ipynb
```

Add `--with-short-training` to test the training cell in the executed copy. This does not enable training in the source notebook.

## 3. Run a fixed engineering design and the baselines

```powershell
.\.venv\Scripts\python.exe scripts/inspect_environment.py --seeds 103 --out runs/my_fixed_design
.\.venv\Scripts\python.exe scripts/run_baselines.py --seeds 101 102 103 104 105 --out runs/my_baselines
```

`inspect` uses the fixed rule: mix, set 370 K, react at parameter 0.85, separate at 0.9, collect the rich outlet, purge the residual. Feed sampling is reproducible for a given seed. To remove feed variation altogether, set `environment.feed_variation: 0.0` in a copied YAML configuration and pass `--config`.

The output contains `metadata.json`, comparison files, and a directory for each policy with `episodes.csv`, `summary.json`, `designs.json` and `best_design.png/json`. The notebook displays component stream tables using `flowsheet_rl.inspection.stream_table`; the complete stream data are also stored on graph edges in the JSON designs.

## 4. Generate and evaluate designs with the saved policy

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint runs/my_quick_training/checkpoint.pt --seeds 101 102 103 104 105 --out runs/my_checkpoint_evaluation
```

This runs deterministic inference, with categorical argmax and Beta means. The checkpoint's environment configuration is loaded automatically. Inspect `evaluation/best_design.png`, `evaluation/designs.json` and `evaluation/episodes.csv` together: higher return alone does not imply feasibility. For the existing checkpoint, purging both feeds is the observed behavior.

Evaluation does not change weights. Repeating the same deterministic cases checks reproducibility; it is not a new independent training replicate. Choose additional held-out feed seeds with `--seeds` for broader testing.

## 5. Start a small training run

```powershell
.\.venv\Scripts\python.exe scripts/train.py --config configs/train.yaml --updates 2 --rollout-steps 32 --out runs/my_quick_training
```

This collects 64 transitions, not 64 episodes. It checks your setup quickly. One complete historical 256-transition run took about 9.17 seconds inside the training routine on the recorded CPU setup; startup, installation and report generation are additional. Training time depends on graph lengths and machine load, so measure your own pilot before planning hours of execution.

For a first learning pilot:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --config configs/train_pilot.yaml --out runs/my_pilot
```

The pilot requests 40 updates × 256 transitions = **10,240 transitions**, with checkpoints and evaluation every five updates. This budget is provided for you to run; it has not been completed as part of the report.

## 6. Monitor, interrupt, and continue

The terminal prints update, transitions, episode count, loss and the first feasible training step. From a second PowerShell terminal in the same project directory:

```powershell
Get-Content runs/my_pilot/progress.json
Get-Content runs/my_pilot/training.json -Tail 30
```

| File | Meaning |
|---|---|
| `metadata.json` | Configuration, source hashes, packages and invocation |
| `progress.json` | Latest completed update; may be ahead of the saved checkpoint |
| `checkpoint.pt` | Latest completed saved update, including optimizer and rollout state |
| `training.json` | Training episodes, update diagnostics and periodic evaluation, persisted at checkpoints |
| `evaluation/` | Final deterministic evaluation, created after normal completion |

Use Ctrl+C to stop. The last saved checkpoint remains available; work since that save must be repeated. With an interval of five, up to five updates may need repeating if interrupted during a save. Saves use a temporary file and atomic replacement. Only the latest checkpoint is retained within a run, not a complete checkpoint archive.

Continue in a **new directory**:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --resume runs/my_pilot/checkpoint.pt --updates 400 --out runs/my_pilot_continued
```

`--updates 400` is the **total target**, including updates already completed. With 256 transitions per update, it targets 102,400 cumulative transitions. Configurations and evaluation seeds are inherited from the checkpoint when omitted. Keep the same environment, model, reward and optimizer settings for exact continuation. The total update budget and saving/evaluation intervals can change; incompatible settings are rejected. Use the same code and locked environment for reproducibility.

New checkpoints preserve the partially built flowsheet, episode accounting, optimizer, Torch RNG, feed RNG and minibatch RNG. A split run was tested against an uninterrupted run and produced identical weights and episode records. Continuation starts from an update boundary, not the exact interrupted instruction or minibatch.

The original September 21 smoke checkpoint predates this support. Use it as **starting weights for a new run**, rather than exact continuation:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --initialize-from runs/my_quick_training/checkpoint.pt --updates 40 --rollout-steps 256 --out runs/my_warm_start
```

This resets optimizer and rollout history and inherits the original configuration unless you supply another. Its old minibatch size remains 16; the pilot YAML uses 32. Label this as a warm start in comparisons. Starting from scratch is also a useful control: the early policy has no demonstrated engineering advantage.

## 7. Train longer and compare independent seeds

For a fresh 102,400-transition run:

```powershell
.\.venv\Scripts\python.exe scripts/train.py --config configs/train_long.yaml --out runs/my_long_run
```

This uses 400 updates, rollout length 256, two epochs, minibatches of 32, checkpoints every five updates and deterministic evaluation every ten. It uses CPU and one Torch thread; it does not use parallel environments or automatically stop on convergence. Keep the machine awake and the terminal open. No long campaign has been launched for you.

After the run completes, generate the learning and final-evaluation plots with:

```powershell
.\.venv\Scripts\python.exe scripts/plot_training_results.py --run runs/my_long_run
```

This writes `training_overview.png` and `evaluation_by_seed.png` under `runs/my_long_run/plots`. The final flowsheet is already stored as `evaluation/best_design.png`.

### Unattended multi-seed campaign

The campaign runner executes independent seeds sequentially, evaluates every final checkpoint on the same feed cases, runs the baselines once, and writes `campaign_summary.json`, `campaign_summary.csv`, and `learning_curves.png`. The default long campaign requests three seeds × 400 updates × 256 transitions = **307,200 transitions** in total.

Run it in the foreground first if you want to see every update:

```powershell
.\.venv\Scripts\python.exe scripts/run_training_campaign.py --config configs/train_long.yaml --training-seeds 7 11 19
```

To leave it running in a hidden background process on Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_training_campaign.ps1 -Profile long
```

The launcher prints the process ID, campaign directory, controller logs and PID file. It prefers the project's `.venv`; supply `-PythonPath` only when intentionally using another compatible environment. While the Python controller is active, it asks Windows to keep the system awake; the display may still turn off. A larger budget can be requested without editing YAML:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_training_campaign.ps1 -Profile long -Updates 800 -TrainingSeeds 7,11,19
```

Monitor without modifying the run:

```powershell
Get-Content runs/long_campaigns/<campaign>/campaign.json
Get-Content runs/long_campaigns/<campaign>/seed_0007/progress.json
Get-Content runs/long_campaigns/<campaign>/campaign_summary.json
```

If Windows, VS Code or the process stops, launch recovery into a new continuation segment:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_training_campaign.ps1 -ResumeCampaign runs/long_campaigns/<campaign>
```

Completed seeds are skipped. An incomplete seed resumes from its last durable checkpoint in a new `seed_####_resume_##` directory. At most the updates since the last checkpoint are repeated. Do not run two controllers against the same campaign directory.

Estimate remaining time from several pilot updates: remaining transitions divided by measured transitions per second, allowing extra time for evaluations and larger flowsheets. Do not interpret a specific budget as sufficient before inspecting feasibility and behavior.

For independent initializations, the following runs three pilots **sequentially**:

```powershell
foreach ($runSeed in @(7, 11, 19)) {
    .\.venv\Scripts\python.exe scripts/train.py --config configs/train_pilot.yaml --seed $runSeed --seeds 101 102 103 104 105 --out "runs/pilot_seed_$runSeed"
    if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $runSeed" }
}
```

Compare feasible fraction, product purity/recovery, utilities, equipment count, termination reasons and actual flowsheets, alongside return. Keep the hold-out feeds fixed for comparisons; reserve a second set of seeds for final testing rather than repeatedly tuning on every case. Report variation between independent training seeds separately from variation across feed cases. Five feed evaluations of one checkpoint are not five training runs.

## 8. Change settings deliberately

Copy a configuration, edit it, then run with `--config configs/my_case.yaml`. See [configuration definitions](../src/flowsheet_rl/config.py) for allowed fields. Existing YAML files can omit fields and use validated defaults.

```yaml
environment:
  feed_a: 1.0
  feed_b: 1.0
  feed_variation: 0.1
  feed_temperature: 300.0
  purity_min: 0.9
  recovery_min: 0.6
  fidelity: source
  max_units: 8
  max_steps: 16
train:
  seed: 7
  updates: 40
  rollout_steps: 256
  epochs: 2
  minibatch_size: 32
  checkpoint_interval: 5
  evaluation_interval: 5
```

Changing feeds within the original range can be evaluated immediately; larger shifts require new validation and may require retraining. Changing rewards, specification limits or fidelity changes the task: use a new run or documented fine-tuning, not exact resume. Specification thresholds and fidelity are not explicit observation features, so one checkpoint should not be assumed to condition on arbitrary changes to them.

Changing species, feature dimensions, unit names/order, action heads or encoder dimensions is an implementation change, with compatibility checks, tests and new training required. No configuration switch adds recycle or rigorous thermodynamics.

## 9. Inspect the compact recorded result

```powershell
Get-Content examples/long_run/progress.json
Get-Content examples/long_run/evaluation/summary.json
```

The mini repository retains the completed run's metadata, final evaluation, designs, and plots. It omits its checkpoint and full training history. Read [the results page](results.md) for the evidence boundary and interpretation, or run the long configuration to create a complete new record.

If a command says the output already exists, change `--out`; do not delete historical results to rerun it. If exact resume rejects a legacy checkpoint, use `--initialize-from`. If notebook imports fail, select `.venv/Scripts/python.exe` and rerun installation with the notebook extra.
