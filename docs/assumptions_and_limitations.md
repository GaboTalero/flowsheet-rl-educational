# Assumptions, evidence and limits

This is an architecture-level educational reproduction of a computational workflow,
not a reproduction of the paper's process model, optimization results or performance.

Implemented: sequential directed graph construction, explicit undefined/open
boundaries, masked discrete hierarchy, unit-conditional Beta variables, graph critic,
PPO/GAE, analytical simulator, random/rule baselines, seeded evaluation, checkpointing,
and two-fidelity transfer experiments. See paper_mapping.md for verified source
concepts and differences. The package was built without author source code or
supplementary hyperparameter claims.

Deferred: recycle and tear-stream convergence; rigorous thermodynamics; equilibrium
chemistry/distillation; real economics; simulator adapters; continuous vector
parameters; general multi-port unit libraries; distributed training; MINLP; LLMs;
beam/enumerative search; isomorphism-canonical topology counts and structural
diversity distances. The ID-based graph serialization is reproducible for replay
but is not advertised as a canonical representation under arbitrary graph isomorphism.

The synthetic target changes kinetics, separation and utility costs. It is a
controlled distribution shift, not a higher-accuracy physical model. Transfer
strategies copy all matching weights, copy only the encoder, or freeze a copied
encoder before unfreezing. Nonmatching non-encoder heads remain freshly initialized
and are logged; incompatible encoders and semantic schemas are rejected. Zero-shot
target performance is measured at zero target steps. Source and target training
budgets are separately logged; transfer is not described as having the same total
training cost as scratch.

Default evaluation varies feed rates within the same bounds using seeds distinct
from training seeds. It is held-out cases, not a large out-of-distribution benchmark.
The short experiment uses only two training seeds and five evaluation cases. Standard
deviations describe this small sample; no statistical superiority claim is justified.
A short PPO run only verifies finite losses, gradients, parameter changes, checkpoint
loading and episode execution. It may never discover a feasible design.

Exploration is difficult because early STOP/purge choices prevent reactants from
meeting. The rule baseline establishes reachability independently of RL.
Increasing the reward alone is not evidence of engineering quality: inspect purity,
recovery, closure, costs and termination reasons together. Masks encode known
impossibilities (e.g. no reactant pair, occupied ports, missing C for separation);
they deliberately permit many low-value but physically valid actions.

The directed encoder receives incoming messages only. Its finite depth limits
long-range local context; the global pooled embedding summarizes the full graph.
The small Python loop over each graph's hierarchy favors clarity over throughput.

Gymnasium's checker can warn that the graph feature Boxes have infinite upper bounds;
actual emitted values are required to be finite. PyG with the locked Python/PyTorch
versions may issue upstream deprecation warnings. Neither warning indicates a failed
balance or failed smoke test. CPU deterministic replay is tested on this locked
environment; bitwise equality across platforms or dependency versions is not promised.

Next milestone: run a preregistered multi-seed learning study with larger but matched
budgets, retained hold-out feed cases, random/rule/untrained controls, and failure-mode
analysis. If sparse exploration remains the bottleneck, test one transparent
curriculum or rule-demonstration warm start as a separately labeled ablation.
