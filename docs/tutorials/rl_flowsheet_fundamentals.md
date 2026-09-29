# Deep Reinforcement Learning for Flowsheet Synthesis – Modules 1 and 2

## Overview

This report develops the first two modules of a learning journey toward “Deep reinforcement learning for chemical-process flowsheet synthesis using graph neural networks,” targeted at a process-systems engineer with prior experience in conventional process design and applied machine learning but limited formal background in reinforcement learning (RL), Markov decision processes (MDPs), and graph representations of flowsheets.
It is motivated by recent work that uses hierarchical actor–critic RL with graph neural networks (GNNs) to generate economically viable flowsheets directly by interacting with process simulators.

The primary orientation source is the talk “Reinforcement learning for the design of chemical processes using graph neural networks” given by Qinghe Gao in the Process Intelligence Research group at TU Delft, which summarizes work on flowsheet generation via hierarchical RL and GNNs and related transfer-learning ideas.
The underlying main paper is “Flowsheet generation through hierarchical reinforcement learning and graph neural networks” by Stops, Leenhouts, Gao, and Schweidtmann, published in AIChE Journal, 69(1), e17938 (2023), with DOI 10.1002/aic.17938.
A complementary review is “Deep reinforcement learning for process design: Review and perspective” by Gao and Schweidtmann, Current Opinion in Chemical Engineering, 44 (2024), article 101012, DOI 10.1016/j.coche.2024.101012.

Module 1 explains reinforcement learning as a process-synthesis problem, with careful MDP (and POMDP) formulations tailored to flowsheet design.
Module 2 explains how a process flowsheet can be represented as a directed heterogeneous graph suitable for downstream use with GNN-based RL agents.

***

## Module 1 – Reinforcement Learning as a Process-Synthesis Problem

### 1.1 Intuitive view: RL vs. supervised learning, optimization, and control

At an intuitive level, reinforcement learning studies how an agent can learn to make a sequence of decisions based on trial and error, guided by feedback in the form of rewards.
This is conceptually similar to how a human process engineer iteratively builds and refines a flowsheet in a simulator: choose the next unit, reconnect streams, adjust operating variables, re-simulate, and judge the design based on performance indicators.

In supervised learning, the task is to learn a mapping from inputs to outputs given labeled examples, such as predicting the product purity from a fixed flowsheet and operating conditions.
There is no notion of the learner’s actions changing the environment; instead, the learner passively fits a function to static data.

Conventional mathematical optimization, including MINLP-based superstructure optimization, abstracts design into static decision variables and constraints, and then uses algorithms to minimize a cost or maximize an objective.
Although sequential algorithms like branch-and-bound explore multiple candidate designs, the sequence itself is not modeled explicitly, and intermediate choices are typically not evaluated by a learned policy.

Classical process control focuses on keeping a process near desired operating conditions in real time, often using feedback controllers that respond to measured disturbances.
In contrast, RL for process synthesis is about choosing the structure and operating conditions of a process before it exists, using repeated simulations rather than on-line measurements.

In RL for flowsheet synthesis:
- The **agent** is an artificial process designer that proposes modifications to the flowsheet.
- The **environment** is a process simulator that evaluates the current flowsheet and returns performance measures and convergence status.
- **Rewards** encode how good the design is, e.g., net present value, purity, utility cost, or feasibility penalties.
- The agent learns a **policy**: a rule mapping its information about the current flowsheet to design actions.

### 1.2 Core RL terminology for flowsheet synthesis

RL is commonly formalized using Markov decision processes (MDPs).
Key concepts include:

- **Agent**: the decision-maker that selects actions based on information about the current situation.
- **Environment**: everything external that responds to actions and produces new situations and rewards.
- **State** $s$: a mathematical representation of the environment’s situation at a given time step.
- **Observation** $o$: the information actually available to the agent about the state; in fully observable problems, $o = s$.
- **Action** $a$: a decision made by the agent, such as adding a unit, connecting streams, or choosing operating conditions.
- **Transition**: the probabilistic or deterministic rule that maps $s_t$ and $a_t$ to the next state $s_{t+1}$.
- **Reward** $r_t$: a scalar feedback signal quantifying the immediate utility of the transition at time step $t$.
- **Policy** $\pi(a|h_t)$: the agent’s rule for choosing actions as a function of its information; for MDPs, $\pi(a|s_t)$ depends only on the current state.
- **Trajectory**: a sequence of states, actions, and rewards $(s_0, a_0, r_0, s_1, a_1, r_1, \dots)$.
- **Episode**: a finite trajectory that starts in some initial state and ends when the process reaches a terminal condition.
- **Terminal state**: a state in which no further actions are taken, such as “flowsheet complete” or “design rejected.”
- **Return**: the cumulative reward over an episode, possibly discounted in time.

In flowsheet synthesis, a typical step might be:
- Agent selects an open material stream and a unit operation type (discrete decisions), and then chooses design variables such as stage numbers and operating pressures (continuous decisions).
- The simulator re-solves the flowsheet, returning updated stream properties and convergence flags.
- Rewards indicate whether the new flowsheet is feasible and how it performs economically or environmentally.

### 1.3 The Markov property and why it matters

The Markov property states that the future evolution of the system depends on the past only through the current state, not on the full history.
Formally, an MDP assumes that
$$
\Pr(s_{t+1} \mid s_0, a_0, \dots, s_t, a_t) = \Pr(s_{t+1} \mid s_t, a_t), \quad \forall t.
$$
This property is crucial because most RL algorithms, including temporal-difference (TD) methods and actor–critic architectures, rely on the assumption that the value of a state can be defined as a function of that state alone.

For flowsheet synthesis, the Markov property requires that the state representation captures all information needed for predicting the impact of any admissible next design action on future rewards.
If, for example, the state includes the full topology of the flowsheet, all relevant stream properties, and convergence and design status, then adding a new unit and re-simulating depends only on this information and the chosen action.
If some hidden variables, such as yet-unresolved recycle convergence or unmodeled dynamic effects, influence future performance but are not included in the state, the problem becomes partially observable.

### 1.4 States vs. observations and partial observability in process synthesis

A **state** is a mathematical abstraction representing everything needed to determine probabilistic transitions and rewards.
An **observation** is what the agent can see; in partially observable systems, observations do not contain all state information.

In process synthesis with simulators, several sources of partial observability can arise:
- The simulator may provide convergence flags without revealing underlying residuals or solution basin information.
- Certain thermodynamic or kinetic parameters may be uncertain or estimated from prior models, not directly visible in the state.
- Long-term stability, controllability, or exergy efficiency may not be computable until the flowsheet is more complete.

Formally, a partially observable MDP (POMDP) introduces an observation $o_t$ drawn from an observation model $O(o_t \mid s_t, a_{t-1})$.
The agent’s policy becomes $\pi(a_t \mid h_t)$, where $h_t$ is the history of past observations and actions, or an internal belief state summarizing them.

For flowsheet synthesis, treating the problem as a POMDP may be appropriate when:
- Simulators provide noisy or partial output, e.g., approximate shortcut results for distillation.
- Only some performance indicators are available during intermediate steps, with others evaluated only at the end.
- The agent must reason about unobserved aspects such as future heat-integration potential or dynamic controllability.

### 1.5 Deterministic vs. stochastic transitions in simulators

In principle, a deterministic steady-state process simulator maps a flowsheet topology and operating conditions to a unique set of stream properties and unit outcomes.
Thus, the transition from state $s_t$ to $s_{t+1}$ after an action $a_t$ (e.g., adding a unit) is deterministic given the simulation model.

However, several sources of stochasticity may appear in RL formulations:
- Numerical solvers may fail or converge to different solutions depending on initial guesses.
- The environment may randomize feed conditions, model parameters, or disturbance scenarios to improve policy robustness.
- Approximations such as shortcut simulators or surrogate models may introduce modeling errors that are effectively stochastic from the agent’s perspective.

Therefore, RL formulations for flowsheet synthesis should allow for stochastic transitions and design reward functions and policies accordingly.

### 1.6 Finite-horizon episodic formulations in flowsheet synthesis

Flowsheet synthesis naturally fits finite-horizon episodic RL formulations:
- Each episode starts from an initial design (often an empty flowsheet with specified feeds).
- The agent iteratively adds units, connects streams, and sets operating conditions.
- The episode terminates when a stopping criterion is met: flowsheet complete, maximum number of units, or infeasible/failed design.

A finite horizon $T$ can be defined in terms of maximum design steps, maximum units, or maximum simulation calls.
The return is then
$$
G_0 = \sum_{t=0}^{T-1} \gamma^t r_t,
$$
where $\gamma \in [0, 1]$ is a discount factor.
This framing supports hierarchical action spaces in which each macro-step (e.g., “add a column”) may encapsulate multiple micro-steps (e.g., select side-draw, reflux ratio).

### 1.7 Policies, value functions, and advantage functions

In RL, the goal is to learn a **policy** $\pi(a|s)$ that maximizes expected return.
Two central value functions are:
- **State-value function** $V^{\pi}(s)$: expected return starting from state $s$ and following policy $\pi$.
- **Action-value function** $Q^{\pi}(s, a)$: expected return starting from state $s$, taking action $a$, and then following $\pi$.

The **advantage function** $A^{\pi}(s, a)$ measures how much better taking action $a$ is than following the average action under $\pi$ in state $s$:
$$
A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s).
$$
Actor–critic algorithms often use function approximators (neural networks, including GNNs) to estimate $V^{\pi}$, $Q^{\pi}$, or $A^{\pi}$ and update the policy parameters based on advantage estimates.

In flowsheet synthesis with hierarchical hybrid actions, the policy may be factored into discrete decisions (select stream, select unit type) and continuous decisions (operating variables), while the critic evaluates the full flowsheet graph to estimate $V^{\pi}(s)$.

### 1.8 Discounted vs. undiscounted returns

Discounting captures the idea that immediate rewards are often more important than distant rewards.
In many episodic design problems, especially when rewards are given only at the end of an episode, discounting is less critical, and an undiscounted return $G_0 = \sum_{t=0}^{T-1} r_t$ can be used.

For flowsheet synthesis, several choices are possible:
- **Undiscounted terminal reward**: evaluate only the final flowsheet (e.g., net present value), treating all intermediate rewards as zero.
- **Intermediate rewards with discounting**: reward convergence, feasibility, and incremental performance improvements, with discounting to encourage faster achievement of good designs.

Choosing $\gamma < 1$ can help stabilize learning and ensure that value functions remain bounded even if episodes are long.

### 1.9 Bellman expectation and optimality equations

The Bellman expectation equations relate value functions to one-step transitions under a fixed policy $\pi$.
For the state-value function:
$$
V^{\pi}(s) = \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ \mathbb{E}_{s' \sim P(\cdot|s,a)} [ r(s,a,s') + \gamma V^{\pi}(s') ] \right],
$$
where $P(s'|s,a)$ is the transition probability and $r(s,a,s')$ is the expected immediate reward.

For the action-value function:
$$
Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P(\cdot|s,a)} [ r(s,a,s') + \gamma \mathbb{E}_{a' \sim \pi(\cdot|s')} [Q^{\pi}(s', a')] ].
$$

The Bellman optimality equations characterize the optimal value functions $V^{*}$ and $Q^{*}$ for the best possible policy:
$$
V^{*}(s) = \max_{a} \mathbb{E}_{s' \sim P(\cdot|s,a)} [ r(s,a,s') + \gamma V^{*}(s') ],
$$
$$
Q^{*}(s, a) = \mathbb{E}_{s' \sim P(\cdot|s,a)} [ r(s,a,s') + \gamma \max_{a'} Q^{*}(s', a') ].
$$
These recursive equations underlie dynamic programming, value iteration, and Q-learning.

In flowsheet synthesis, solving the Bellman optimality equations exactly is infeasible because the state space (all possible flowsheets and operating conditions) and action space (all possible modifications) are enormous and continuous.
Instead, deep RL uses approximate methods that sample transitions from the simulator and fit parametric models to approximate value functions and policies.

### 1.10 Temporal-difference learning: conceptual role

Temporal-difference (TD) learning updates value estimates based on the difference between predicted and realized returns over a single time step.
For example, a simple TD(0) update for $V^{\pi}$ is:
$$
V(s_t) \leftarrow V(s_t) + \alpha \left( r_t + \gamma V(s_{t+1}) - V(s_t) \right),
$$
where $\alpha$ is a learning rate.

Conceptually, TD methods allow value functions to be learned from incomplete trajectories without waiting for full episode returns.
This is critical in design settings where episodes may be long and expensive (e.g., many simulation calls) and where intermediate rewards carry meaningful information.

Actor–critic methods often use TD-based updates for the critic and policy-gradient updates for the actor, combining the stability of value-based bootstrapping with the flexibility of function approximation.
Proximal policy optimization (PPO), used in the flowsheet-generation work, employs clipped policy updates and TD-based advantage estimation to stabilize learning in high-dimensional continuous and hybrid action spaces.

### 1.11 Exploration vs. exploitation in design

RL agents must balance **exploration** (trying new actions to discover potentially better designs) and **exploitation** (using known good actions to obtain high rewards).
Common mechanisms include:
- Stochastic policies, where action probabilities reflect uncertainty.
- Entropy bonuses in the objective function to encourage diversity.
- Exploration noise in continuous actions.

In flowsheet synthesis, exploration is analogous to trying unconventional unit sequences or recycle structures that may initially seem suboptimal but ultimately yield better performance.
Exploitation corresponds to reinforcing patterns that have historically led to high net cash flow or exergy efficiency.
Because process simulators are computationally expensive, exploration must be efficient; hierarchical RL and transfer learning from shortcut simulators help reduce the cost of exploring large design spaces.

### 1.12 Sparse terminal rewards vs. intermediate reward shaping

A purely terminal reward scheme evaluates only the final flowsheet, such as net cash flow or overall separation performance.
This leads to **sparse rewards**: most intermediate actions yield zero reward, making learning difficult because it is hard to assign credit or blame to specific design choices.

Reward shaping introduces intermediate rewards that provide guidance during the design process, for example:
- Positive reward when the flowsheet converges after adding a unit.
- Negative reward when convergence fails or when physical constraints are violated.
- Incremental rewards for improvements in product purity or utility cost.

These signals help TD learning and actor–critic methods propagate information backward through the episode so that the policy learns which local design decisions tend to lead to good final designs.
However, reward shaping must be designed carefully to avoid changing the optimal policy or inadvertently encouraging undesirable behaviors.

### 1.13 Why RL does not automatically guarantee global optimality

Although RL aims to find high-performing policies, it does not generally guarantee global optimality in continuous, high-dimensional design spaces.
Reasons include:
- Function approximation error: neural networks may not perfectly represent the optimal value functions or policies.
- Limited exploration: practical training campaigns explore only a subset of possible flowsheets.
- Non-convex optimization: training the networks involves non-convex objectives with local minima.
- Approximate algorithms: PPO and related methods maximize surrogate objectives that approximate but do not equal the true expected return.

Therefore, while RL-based flowsheet generators can discover creative and economically competitive designs, they should be viewed as tools for generating good candidates rather than global optimizers, especially in the presence of complex thermodynamics and process interactions.

### 1.14 Why “superstructure-free” does not mean “free of predefined engineering choices”

RL-based flowsheet synthesis is sometimes described as “superstructure-free” because it does not require a fully enumerated superstructure model of all possible flowsheets.
Instead, the agent builds flowsheets sequentially by adding units and connections, guided by learned policies.

However, this does not mean that the formulation is free of engineering choices:
- The **action space** must still specify which unit types and connections are allowed.
- The **state representation** must choose which variables to include, such as stream compositions and temperatures.
- The **reward function** must embody priorities like economic performance, exergy efficiency, or emissions.
- The **simulator** must encode thermodynamic models and unit-operation models.

Thus, “superstructure-free” means that the precise topology is not hard-coded upfront, but the design space is still constrained by modeling choices that reflect engineering knowledge and design preferences.

### 1.15 MDP and POMDP formulations for flowsheet synthesis

This section formalizes flowsheet synthesis as an MDP or, where appropriate, a POMDP.
The formulations are generic and can be specialized to different simulators and objectives.

#### 1.15.1 MDP formulation

An MDP for flowsheet synthesis can be defined as a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0, T)$:
- $\mathcal{S}$: state space of flowsheets and associated variables.
- $\mathcal{A}$: action space of design modifications.
- $P(s'|s,a)$: transition kernel induced by the simulator.
- $R(s,a,s')$: reward function.
- $\gamma \in [0,1]$: discount factor.
- $\rho_0$: initial state distribution.
- $T$: maximum episode length.

**State space $\mathcal{S}$**
Each state $s$ encodes a partial or complete flowsheet:
- Graph $G = (V, E)$ with nodes representing unit operations and feeds/products, and edges representing material streams.
- Node features such as unit type, design status, operating conditions, and convergence flags.
- Edge features such as flowrate, composition vectors, temperature, pressure, and phase.
- Global features such as total utility consumption, total capital cost estimates, and convergence status of the entire flowsheet.

**Action space $\mathcal{A}$**
Actions are hierarchical and hybrid:
- Discrete decisions: select an open stream; select a unit type to attach; choose which port to connect (e.g., feed, product, utility).
- Continuous decisions: specify design variables (e.g., number of stages, reflux ratio, column pressure, reactor temperature).

An action $a$ can be represented as a tuple $(a^{\text{stream}}, a^{\text{unit}}, a^{\text{design}})$ with discrete components $a^{\text{stream}}, a^{\text{unit}}$ and continuous vector $a^{\text{design}}$.

**Transition kernel $P$**
Given $s_t$ and $a_t$, the simulator constructs a new flowsheet by adding or modifying units and streams, then solves the material and energy balances and unit models.
The next state $s_{t+1}$ contains updated graph topology, stream properties, and convergence and feasibility flags.
If the simulator is deterministic, $P(s_{t+1}|s_t,a_t)$ is a Dirac measure; if randomization or numerical issues occur, $P$ may be stochastic.

**Reward function $R$**
Rewards combine intermediate and terminal components:
- Intermediate reward $r_t^{\text{int}}$ for convergence success, violation of physical constraints, or incremental improvement in performance indicators.
- Terminal reward $r_T^{\text{term}}$ based on economic or multi-objective metrics such as net cash flow, exergy efficiency, or emissions.

The total reward at time $t$ can be written as $r_t = r_t^{\text{int}} + \mathbb{1}_{t=T} r_T^{\text{term}}$.

**Episodes and terminal states**
Episodes terminate when:
- A design-completion condition is met (e.g., all feeds and products attached, no open material streams).
- A maximum number of units or actions is reached.
- The simulator fails irrecoverably (e.g., sustained non-convergence).

Terminal states encode a completed flowsheet or a failed design.

#### 1.15.2 POMDP formulation

When the state cannot be fully observed, the problem becomes a POMDP characterized by $(\mathcal{S}, \mathcal{A}, P, R, \gamma, \mathcal{O}, O, \rho_0, T)$:
- $\mathcal{O}$: observation space.
- $O(o|s,a)$: observation model.

Observations $o_t$ may include only a subset of variables, for example:
- Shortcut simulation results without detailed stage-by-stage profiles.
- Aggregated cost estimates without full breakdown.
- Feasibility and convergence flags without residuals.

The agent maintains an internal belief or recurrent hidden state $b_t$ that summarizes past observations and actions.
Policies become $\pi(a|b_t)$, and critics may be conditioned on $b_t$ or on the graph plus additional memory-based features.
This formulation is particularly relevant for multi-fidelity setups in which the agent is pre-trained on a shortcut simulator and fine-tuned on a rigorous simulator, as in transfer-learning strategies for process synthesis.

***

## Module 2 – Representing a Flowsheet as a Directed Heterogeneous Graph

### 2.1 Why flowsheets should be represented as graphs

A process flowsheet is a network of unit operations connected by material (and potentially energy) streams.
Each unit can have multiple inlets and outlets, and streams can be split, mixed, and recycled.
This naturally corresponds to a graph: units and feeds/products are vertices, and streams and connections are directed edges.

Alternative representations such as images, sequences, or fixed-length vectors are less suitable:
- Images of P&IDs or flowsheets are visually intuitive but discard explicit topology and numerical attributes and are not invariant to layout choices.
- Sequences of units ignore branching and recycling unless complex serialization schemes are used.
- Fixed-length vectors require a priori limits on the number of units and streams and struggle with variable-size designs.

Graphs support variable-size, combinatorial structures and can encode both discrete topological information and continuous process variables, making them a natural basis for GNN-based RL agents.

### 2.2 Basic graph terminology as applied to flowsheets

Relevant graph concepts include:
- **Vertex (node)**: an entity such as a unit operation, feed, product, or utility source.
- **Edge**: a directed connection representing a stream or linkage between nodes.
- **Direction**: orientation of edges indicating flow from upstream to downstream nodes.
- **Path**: a sequence of nodes connected by edges, representing a process route from feed to product.
- **Cycle**: a closed path, corresponding to recycle loops.
- **Neighborhood**: the set of nodes directly connected to a given node.
- **Connectivity**: whether all relevant nodes are connected in a way that allows the process to operate.

Directed graphs capture the causal direction of material flow and help ensure that simulation solvers respect upstream/downstream relationships.

### 2.3 Directed graphs, multigraphs, and heterogeneous graphs

A flowsheet graph is typically:
- **Directed**: edges have a direction from source node to target node.
- **Multigraph**: multiple edges between the same pair of nodes may represent parallel streams or different phases.
- **Heterogeneous**: nodes and edges can be of different types (e.g., reactors, distillation columns, heat exchangers; material streams, utility streams).

Heterogeneity is important because different unit types have different operational constraints and design variables.
A heterogeneous graph representation allows GNNs to treat nodes and edges differently based on their type, improving expressiveness and learning.

### 2.4 Alternative modeling choices: node–edge semantics

Several modeling choices exist for how to map units and streams into nodes and edges.

#### 2.4.1 Units as nodes, material streams as edges

The most common representation in RL-based flowsheet generation treats unit operations as nodes and material streams as directed edges.
Advantages:
- Direct correspondence between units and node features (unit type, operating conditions).
- Stream properties live on edges, naturally representing flow and composition.
- Recycle loops, splitters, and mixers are captured as cycles and branching patterns in the graph.

Disadvantages:
- Complex units with multiple ports may need additional modeling to distinguish inlet and outlet edges.
- Non-material connections (e.g., control signals, heat integration) must be added as additional edge types or separate graphs.

#### 2.4.2 Streams as nodes, connections as edges

An alternative is to represent streams as nodes and connections between streams (e.g., jumping through units) as edges.
Advantages:
- Focus on stream properties and routing, which may be useful when material movement is central.
- Units can be treated as transformations encoded in edge features or as meta-nodes.

Disadvantages:
- Unit-level design variables become harder to attach to nodes.
- Many process-systems engineering tools naturally think in terms of units; this representation may be less intuitive.

#### 2.4.3 Bipartite or factor-graph representations

A bipartite representation can model both units and streams as nodes, with edges connecting streams to units.
This resembles incidence graphs and can be extended to hypergraphs when units join multiple streams.
Advantages:
- Symmetric treatment of units and streams.
- Natural extension to multi-commodity or multi-phase networks.

Disadvantages:
- Graph size and complexity increase, complicating GNN architectures.
- Action spaces may become more complex because decisions span multiple node types.

For hierarchical RL with GNNs, using units as nodes and streams as edges has proven practical and aligns with the design of the flowsheet representation in the hierarchical RL paper.

### 2.5 Special node and edge types in flowsheets

Flowsheets contain several special entities that must be represented explicitly:
- **Feed nodes**: sources of material entering the system, often with fixed composition and flowrate.
- **Product nodes**: sinks where final products leave the system, with specifications on purity, recovery, or flow.
- **Utility streams**: edges or nodes representing energy inputs (steam, cooling water, electricity).
- **Open streams**: material streams that are not yet connected to a downstream unit; these define potential action locations.
- **Ports**: specific inlets and outlets of units, such as feed, distillate, bottoms, side draws.
- **Splitters and mixers**: units that branch or merge streams, leading to graph branching and merging.
- **Recycle loops**: cycles in the graph where downstream streams are routed back upstream.

Action masks in RL can use this structural information to determine which streams are open and which actions (unit types, connections) are feasible at any given step.

### 2.6 How the graph evolves during flowsheet construction

During RL-based flowsheet synthesis, the graph evolves step by step:
1. Initial state: a minimal graph with feed nodes and perhaps product targets but no internal units.
2. At each step:
   - The agent selects an open stream or a connection point.
   - The agent chooses a unit type to add.
   - The agent specifies design variables and ports.
3. The simulator builds the corresponding unit model, connects it in the flowsheet, solves balances, and updates stream properties.
4. The graph is updated:
   - New node added for the unit, with its feature vector.
   - New edges added for streams entering and leaving the unit.
   - Existing edges modified if streams are split or merged.

This sequential graph-growth process is analogous to graph generation in other domains, such as molecular design, and has been successfully modeled using graph-based RL.

### 2.7 Node, edge, and global features

A rich set of features can be attached to nodes, edges, and the overall flowsheet.

**Node features** may include:
- Unit type (categorical, encoded via one-hot vectors or learned embeddings).
- Design status (initialized, converged, infeasible).
- Operating variables (e.g., temperature, pressure, reflux ratio).
- Local performance metrics (e.g., local exergy destruction estimates).

**Edge features** may include:
- Flowrate (mass or molar).
- Composition vector of species.
- Phase information (liquid, vapor, solid).
- Temperature and pressure.

**Global flowsheet features** may include:
- Total utility consumption and cost.
- Total capital cost estimates.
- Overall convergence flags (e.g., all recycles converged).
- Aggregated performance metrics (e.g., net cash flow, exergy efficiency).

These features provide the information needed for GNNs to compute a “flowsheet fingerprint,” a latent representation summarized over the graph that can be fed into actor–critic networks.

### 2.8 Encoding categorical unit types and continuous variables

Categorical variables like unit types (reactor, flash, distillation column, heat exchanger) are typically encoded as one-hot vectors or learned embeddings.
One-hot encoding assigns a binary feature vector with a 1 at the index corresponding to the unit type.
Embeddings map unit types to dense vectors learned during training, allowing the model to capture similarities (e.g., different kinds of columns).

Continuous variables such as flowrates, temperatures, and pressures are kept as real-valued features.
To stabilize training and avoid scale issues, these variables are often normalized, for example by subtracting means and dividing by standard deviations, or by mapping to standardized ranges.

### 2.9 Detailed feature examples for flowsheets

For clarity, consider the following feature sets:

**Node feature vector for a distillation column**:
- One-hot unit type: column.
- Design variables: number of stages, feed stage location, reflux ratio, column pressure.
- Convergence status: binary flag for successful solution.
- Local performance: separation factor, tray efficiency indicators.

**Edge feature vector for a liquid stream**:
- Mass flowrate.
- Temperature and pressure.
- Composition vector $x \in \mathbb{R}^n$ for $n$ chemical species.
- Phase indicator: liquid.

**Global features**:
- Total fuel cost.
- Total cooling water consumption.
- Product purity and recovery.
- Overall net cash flow.

These features must be computed consistently by the simulator and stored in the graph at each step.

### 2.10 Feature scaling, missing values, masks, and partial flowsheets

Several practical issues arise when encoding features:
- Different variables have different scales (e.g., pressures vs. temperatures vs. flowrates).
- Some variables may be unavailable before a unit is fully specified or before convergence.

Feature scaling normalizes variables to comparable ranges, helping neural networks train efficiently.
Missing values can be handled by:
- Imputing neutral values (e.g., zeros) with companion masks indicating whether a variable is known.
- Using masking mechanisms in GNNs to avoid propagating information from undefined features.

In partial flowsheets, many nodes and edges may lack complete information.
Proper masking ensures that the GNN-based agent does not rely on variables that are not known at the current design step, reducing information leakage and helping maintain the Markov property.

### 2.11 Adjacency and incidence matrices (to the needed depth)

A graph can be represented algebraically by matrices:
- **Adjacency matrix** $A$, where $A_{ij} = 1$ (or a weight) if there is an edge from node $i$ to node $j$, and 0 otherwise.
- **Incidence matrix** $B$, where each column represents an edge, with non-zero entries indicating its source and target nodes.

These matrices play a role in many GNN formulations and allow compact representation of connectivity.
In practice, flowsheet graphs are often represented using adjacency lists or specialized data structures (e.g., PyTorch Geometric’s `Data` objects) that encode node and edge indices and features.
For RL with GNNs, the key requirement is to have efficient access to node neighborhoods and global graph structure, which these representations provide.

### 2.12 Variable graph size and permutation invariance

Flowsheet graphs have variable size: different designs may contain different numbers of units and streams.
GNNs naturally handle variable-size graphs by applying shared message-passing operations over all nodes and edges.

Permutation invariance means that the graph representation (and the GNN outputs) should not depend on arbitrary node ordering.
GNNs achieve this by aggregating messages using permutation-invariant operations such as summation, averaging, or max-pooling; the flowsheet fingerprint thus depends only on the topology and features, not on indexing choices.

### 2.13 Topological validity vs. physical feasibility

Two distinct notions of validity arise:
- **Topological validity**: the graph respects basic structural constraints, such as having feed nodes and product nodes connected through some path, and no illegal connections (e.g., connecting a product back to a utility source).
- **Physical feasibility**: the flowsheet satisfies material and energy balances, thermodynamic constraints, and unit-model requirements, yielding converged and realistic solutions.

RL agents should be constrained to generate topologically valid graphs through action masks and environment checks.
Physical feasibility is enforced by the simulator; infeasible graphs yield negative rewards and possibly termination.

### 2.14 Action masks and identification of valid next actions

Action masks indicate which actions are valid in a given state, such as which open streams may receive new units or which unit types are permissible given current conditions.
In maskable actor–critic algorithms, the policy network produces logits for all possible actions, and the mask zeroes out invalid ones before sampling.

For flowsheet synthesis, masks can be derived by analyzing the graph:
- Identify open streams that are not connected to downstream units.
- Check unit-specific constraints (e.g., columns require appropriate feed phase and non-zero flow).
- Ensure that adding a unit does not create forbidden cycles or violate topology rules.

Proper masking prevents the agent from generating impossible or nonsensical designs and focuses exploration on feasible regions of the design space.

### 2.15 Markov property and information sufficiency of the representation

To satisfy the Markov property, the graph-based representation must contain all information necessary to predict future simulator responses and rewards given any allowed action.
This requires:
- Complete topology (nodes, edges, directions, and special types).
- Relevant thermodynamic and kinetic variables (e.g., composition, temperature, pressure at all streams).
- Convergence status and any numerical flags that affect future simulation behavior.

If critical information is omitted—for example, residuals in recycle convergence, hidden thermodynamic parameters, or latent opportunities for heat integration—the representation may be partially observable.
Designing the representation thus involves a trade-off between completeness (for Markovity) and tractability (for GNNs and RL training).

### 2.16 Information leakage: avoiding future knowledge at current steps

Information leakage occurs when the state representation includes variables that would only be known in hindsight, such as final net cash flow or performance of units that have not yet been added.
Leakage violates the causal structure and can lead to unrealistic policies that exploit artifacts of the training environment.

For flowsheet synthesis, care must be taken to:
- Include only variables that can be computed at the current partial flowsheet.
- Avoid aggregating performance metrics that depend on future units.
- Ensure that reward shaping is based on information available at the current step.

When multi-fidelity simulators are used, transfer learning should be based on compatible representations; shortcut simulators may compute approximate features, but these features must correspond to quantities that would be available at that stage, not to future detailed simulations.

### 2.17 Representation limitations: recycle loops, heat integration, control structures, and non-material connections

Flowsheet graphs focused on material streams and units may have limitations:
- Long recycle loops with complex convergence behavior may require additional features to capture numerical stability or sensitivity to initial guesses.
- Heat integration and utility networks involve energy flows that may be better represented in separate graphs or as additional edge types.
- Control structures (sensors, controllers, actuators) and information flows are non-material connections that may need distinct graph representations or multi-layer graphs.

These aspects can be integrated by extending the heterogeneous graph to include:
- Energy nodes and edges for heat exchangers and utility connections.
- Control-system nodes and edges for instrumentation and control logic.
- Multi-graphs or multiplex networks that represent different types of interactions (material, energy, information).

However, each extension increases complexity and may require specialized RL and GNN architectures.
The hierarchical RL paper focuses primarily on material flowsheets with unit operations as nodes and material streams as edges, leaving advanced energy-integration and control structures as promising future work.

### 2.18 Summary: information passed to the GNN (for Module 3)

To prepare for GNN-based RL architectures in the next module, it is useful to summarize the information that will be passed to the GNN:
- Graph $G = (V, E)$ with nodes representing unit operations, feeds, and products, and edges representing material streams.
- Node feature matrix $X^{(V)}$, where each row encodes unit type, design variables, convergence status, and local performance indicators.
- Edge feature matrix $X^{(E)}$, where each row encodes flowrate, composition, phase, temperature, pressure, and possibly utility-related attributes.
- Global feature vector $x^{(G)}$ capturing aggregated flowsheet metrics and convergence flags.
- Adjacency or edge-index representation encoding connectivity and direction.
- Masks for missing features and invalid actions (open streams, unit-type constraints).

The GNN will process these inputs through message passing, neighborhood aggregation, and pooling to produce:
- Node embeddings (latent representations of each unit and stream neighborhood).
- A flowsheet fingerprint (graph-level embedding) summarized from node and edge embeddings.

This fingerprint feeds into the actor–critic networks that implement hierarchical hybrid action selection and value estimation, closing the loop between graph representation, RL formulation, and process-systems engineering.

***

---

## References

See the repository's [curated references](../references.md) for stable links to the primary papers, books, and official software documentation cited in this tutorial.
