# Running and interpreting experiments

Every CLI command requires a new output directory and refuses to overwrite an
existing directory. Metadata stores the full configuration and its SHA-256, source
file hashes, exact package versions, Python/platform, and invoked command. This
workspace had no Git repository at implementation time, so no fabricated commit
identifier is recorded. Keep code and lock file with the generated run.

Evaluation emits episode CSV and JSON, summary means and sample standard deviations,
termination-reason counts, complete graph/action records, best design JSON, and a
flowsheet PNG. Baseline mode compares random-valid, rule-based and untrained neural
policies on the same requested seeds. Training saves both its stochastic training
episodes and deterministic hold-out evaluation. Compare the trained evaluation
with the baseline directory from the same configuration and seed list.

Train settings count environment transitions, not episodes: updates*rollout_steps.
All neural initialization and sampling are seeded, with one CPU thread and
deterministic torch algorithms. The random baseline uses its own NumPy generator.
The rule baseline uses the same observations, masks and simulator as the neural agent.

Transfer mode reads configs/transfer.yaml: each training seed gets its own source
pretraining run. Scratch/full/encoder/encoder_frozen variants each receive the same
number of target steps and the same feed-case sequence. The encoder-frozen mode
unfreezes at the configured update. Learning curves include evaluation at target
step zero and after each update. transfer.json retains training and evaluation
records per seed; transfer.png plots performance against target transitions.
Source pretraining costs are additional and explicitly recorded.

Time and episodes to first feasible design are nullable: null means none was found
within the measured budget, not zero time. Evaluation best means highest episode
return among the sampled cases, not a proven optimum. No stochastic evaluation
sampling or long-run learning advantage is inferred from a deterministic smoke run.

## Continuing experiments

See [the run guide](run_model.md) for periodic checkpoints, progress monitoring and
continuation in a new output directory. The supplied pilot and long configurations
request 10,240 and 102,400 transitions respectively; these campaigns have not been
run as part of the report. The [results page](results.md) treats the original
256-transition results as preliminary diagnostics. New logs include separate actor
and critic losses and per-episode engineering metrics; old logs are preserved.
