# Analytical model specification

Every constant below is a synthetic educational **DESIGN-CHOICE**. These are not
experimental measurements or correlations from Stops et al.

## Chemistry and units

Streams carry component molar flows `[F_A,F_B,F_C]` in synthetic mol/s, temperature
in K and pressure in bar. Reaction: `A+B -> C`; stoichiometric vector
`nu = [-1,-1,+1]`. Abstract molecular weights are `[1,1,2]` g/mol. Conserved
elemental inventories are `F_A+F_C` and `F_B+F_C`. Total molar flow is not conserved
by reaction. Every flow must be nonnegative and finite. Temperatures lie in
[280,450] K; pressure is positive and initially 1 bar. No pressure-changing unit exists.

Two feeds contain pure A and pure B at 300 K by default. Each reset independently
samples each rate uniformly within +/-10% of its configured nominal value (1 mol/s).
A seed determines the complete case. Baseline and policy evaluation use identical
seeds; training starts at a separate seed and continues its own random sequence.

Let `p in [0,1]` be the normalized action parameter.

| Unit | Equations | Cost/utility proxy |
|---|---|---|
| Mixer | Sum inlet component flows; T = sum(F_total*T)/sum(F_total); P = min inlet P | Capital 0.1; no utility |
| Thermal | T_out = 280 + 170p; component flows unchanged | Q = 0.01 F_total abs(T_out-T_in); capital 0.3 |
| Reactor | tau = 0.1+3.9p; k = 0.6 exp((T-330)/80); X = min(0.98,1-exp(-k*tau)); xi = X min(F_A,F_B); F_out = F_in + nu xi | Q = 0.1 tau F_total + 0.2 xi; capital 1+0.1 tau |
| Separator | r_C = 0.7+0.28p; leakage l = 0.02+0.08(1-p); rich = F*[l,l,r_C]; residual = F-rich | Q = F_total(0.2+p^2); capital 0.6+0.3p |
| Product | Closes branch; adds component flows to the product pool | No capital or utility |
| Purge | Closes branch; discarded material remains in balance accounting | No sales revenue |
| Junction | Pass-through connector created by a mixer partner connection | No equipment count, cost or utility |

For a zero-flow mixer, temperature is the arithmetic mean of the two inlet
temperatures. Zero-flow composition is [0,0,0]. Pressure averaging, heat of mixing,
phase equilibrium, reaction equilibrium and reaction enthalpy balances are omitted.
Thermal duty can be read as kW using a synthetic equal molar heat capacity of
0.01 kJ/(mol K). Other utility coefficients are arbitrary proxies. The reactor is
isothermal; its duty is a charge for residence and reaction, not a solved energy
balance. The tau variable is a dimensionless residence proxy.

The target simulator uses the same topology and action contracts:
`k_target = 0.8*k/(1+0.15*F_C)`, reactor Q multiplied by 1.2,
`r_C,target = r_C-0.03p^2`, leakage multiplied by 1.2, separator Q multiplied
by 1.3, and thermal Q multiplied by 1.2. Both models are analytical and deterministic.
No stochastic failures or invented commercial-simulator timing claims are included.

## Balances and diagnostics

Units check `sum(F_out)-sum(F_in)-nu*xi`. Whole flowsheets check
`F_feed+nu*sum(xi)-F_frontier`, where the frontier includes all product, purge and
open branches exactly once. Internal edges are never summed as external discharge.
The mass residual is `MW dot (F_feed-F_frontier)`; the two elemental residuals
are also recorded. Default absolute balance tolerance: 1e-9 mol/s (and 1e-9 g/s
for the synthetic mass check). Tests reach tighter residuals.

The directed acyclic graph is evaluated in topological order from the original feeds
after each valid design proposal. Results are recomputed, not accumulated across
simulation calls. A failed proposal is discarded; the last valid graph remains
available and the failure result is returned in info.

Statuses are separate: topology validity, calculation status, numerical convergence
(analytical completion, with no iterative solver), physical feasibility, closure,
product specification, and economic attractiveness. A physically valid but poor
process is not a numerical failure. Injected numerical failures are tested.
Calculation statuses include ok, topology_error, numerical_failure and physical_violation.
An invalid environment action is separately recorded as invalid_action.

Product purity is total routed C divided by all routed product moles. Recovery is
routed C divided by min(total feed A,total feed B). Products are aggregated, so
splitting an identical product into many outlets cannot multiply its credited value.
Specification requires purity >=0.90 and recovery >=0.60. A feasible design also
requires all branches closed and physical balance satisfaction.

The separate profit proxy is `10*recovery - 0.2*Q - 0.3*capital`.
Economic attractiveness requires a feasible design with positive proxy profit.
This is dimensionless, not annualized profit or a currency estimate.

## Reward

Every step pays -0.01. At episode end, the following terms are paid once:

- product_value = 10 * recovery * purity
- utility = -0.2 * total Q
- capital = -0.3 * equipment capital
- complexity = -0.05 * equipment count
- success = +2 if the completed process is feasible and no failure occurred
- specification = -4 * (max(0,0.90-purity)+max(0,0.60-recovery))
- open_streams = -2 * number of unclosed branches
- failure = -10 for invalid actions or simulation failures

All weights are configurable in the reward configuration. A premature STOP is
permitted, but does not turn open material into products. Sinks do not count as
equipment. A cap of eight equipment units still permits branch closure. Sixteen
design steps is the default finite horizon; reaching it returns truncated=True and
the terminal design score. Natural closure, explicit STOP, invalid actions and
calculation failure return terminated=True. Once done, another step requires reset.

Reward-hacking safeguards tested: no free credit for raw feeds, no repeated
intermediate product revenue or success bonuses, no duplicate material through
mixer ports, no credit for unclosed streams, and nonnegative unit costs. Extra units
are still allowed if physically meaningful: whether additional conversion justifies
their cost is the agent's optimization responsibility, not a hidden heuristic mask.

## Independently checkable route

Use equal unit feeds: mix, heat to 370 K (p=90/170), react (p=0.85),
separate (p=0.9), send rich branch to product and residual to purge.
Then tau=3.415, k=0.6*exp(0.5), X=min(0.98,1-exp(-k*tau)),
r_C=0.952 and leakage=0.028. Thus C_product=0.952X,
A_product=B_product=0.028(1-X), recovery=0.952X and
purity=0.952X/[0.952X+0.056(1-X)]. The rule policy uses this route and
evaluates the same equations for the perturbed target.
