# Reference-to-prototype mapping

Primary evidence: the user-supplied Stops, Leenhouts, Gao and Schweidtmann paper,
*Flowsheet generation through hierarchical reinforcement learning and graph neural
networks*, DOI [10.1002/aic.17938](https://doi.org/10.1002/aic.17938), published 2022.
Locators below use the printed article pages (1-14), not a repository cover sheet.
The supplied YouTube URL could not be retrieved; no claims are based on an unseen
transcript. The primary paper takes precedence.

| Reference-work concept | Simplified implementation | Important deviation |
|---|---|---|
| Directed graph, feeds/products/units as nodes; stream attributes on edges; undefined nodes for open streams (section 2.1, p.3, Fig.2) | NetworkX graph and Gym Graph observation converted to PyG Data | New synthetic feature schema and explicit purge/junction nodes |
| Message passing with node and edge features, sum pooling (2.2.2, p.4, Figs.4-5) | Two residual directed message layers and sum readout | PyG instead of authors' DGL; architecture and dimensions are design choices |
| Location -> unit -> continuous design hierarchy (2.2.1, pp.3-4, Fig.3) | Masked open-node choice -> unit -> conditional Beta parameter | Extra STOP option and conditional mixer partner; no recycle |
| Additional GCN for location scoring; selected node ID concatenated with fingerprint (2.2.3, p.4) | Shared node encoder, node-scoring MLP and selected node embedding | No numeric node ID in learned features; preserves reindexing symmetry |
| Individual MLPs per unit emit Beta alpha/beta (2.2.3, p.5) | Separate Beta heads for thermal/reactor/separator | Parameters normalized to [0,1], concentrations >1 for stability |
| Fingerprint MLP critic, PPO clipping, entropy, GAE (section 2 and 2.4, pp.3,5-6) | Transparent PPO using joint action likelihood ratio and exact hierarchical entropy | Joint-ratio loss differs from the paper's sum of individual actor losses; no claim of identical optimizer |
| Terminal net cash flow; negative cash flow divided by ten; zero intermediate reward (2.3, pp.5-6) | Named dimensionless terminal score plus small step cost | No annualization, market prices or negative-reward rescaling |
| Methyl acetate equilibrium reactor, azeotropic distillation, heat exchanger, recycle (2.5-2.6, pp.6-8) | Irreversible A+B->C, analytical recovery splitter, thermal unit, mixer | No real chemistry, phase equilibrium, recycle or nonlinear flowsheet solver |
| Limit of 25 units; remaining open streams sold as products (2.3, p.5) | Eight equipment units by default; explicit sinks; design horizon truncation | Unclosed material is penalized and never silently credited as product |
| Learning performance in paper case study | CPU smoke training/evaluation records | Execution correctness only; numerical results are not reproduced |
| Transfer learning | Source/target nonlinear toy variants, full/encoder transfer and freezing | An extension requested for this project, not attributed to the paper |

The earlier local Gottl/Burger papers describe SynGameZero, hierarchical neural
networks and tree search. Those provide context, but their two-player algorithm is
not the algorithm of Stops et al. The local Burger text identifies itself as
Gottl, Tonges, Grimm and Burger (2021), DOI 10.1002/cite.202100086.

All equations, costs, bounds, masks, hyperparameters and experiment budgets in this
package are **DESIGN-CHOICE** unless explicitly identified above. Paper performance
is **AUTHOR-REPORTED**. Passing project tests are **DEMONSTRATED** for this toy only.
