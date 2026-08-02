# QUBO Rollout Selection

Volt can generate more trajectories than a learner can evaluate in one update.
The QUBO experiment treats allocation as a yes/no decision for each candidate
rollout fragment: retain fragments that appear useful while avoiding a batch
filled with near duplicates.

For candidate fragments (i=1,\ldots,n), let (x_i\in\{0,1\}), normalized
utility (u_i\), and pairwise similarity (s_{ij}\). Selecting approximately
or exactly (k) fragments minimizes

\[
E(x) =
-\lambda_u\sum_i u_i x_i
+\lambda_s\sum_{i<j}s_{ij}x_ix_j
+\lambda_k\left(\sum_i x_i-k\right)^2.
\]

The linear term rewards utility. The quadratic similarity term charges
redundancy. The final quadratic term exposes the target cardinality to an
unconstrained solver. Jörmungandr's built-in classical solver stays in the
exact-cardinality feasible set, constructs a greedy batch, and improves it
with deterministic one-for-one swaps. It emits the full Q matrix so another
classical, simulated-annealing, quantum-inspired, or quantum solver can replace
only the solve step.

This structure follows the utility/diversity formulation explored by
[Quantum-Inspired Episode Selection for Monte Carlo Reinforcement Learning via QUBO Optimization](https://arxiv.org/abs/2601.17570).
Related priority-amplitude ideas appear in
[Quantum-Inspired Experience Replay](https://arxiv.org/abs/2101.02034).
These references motivate an experiment; they do not establish a speed or
sample-efficiency gain for Volt.

## Candidate construction

For transition-mode algorithms, a candidate pool is sampled without
replacement using replay priority. Each candidate embeds the current
observation, state delta, action, reward, and terminal flag.

For trajectory-mode plugins, the binary unit is instead a contiguous fragment:

```text
(actor_id, episode_id, timestep=a..b) -> one QUBO variable
```

Fragments never cross actors, episode identifiers, terminal steps, timestep
gaps, or `rollout_length`. PPO, IMPALA, and APPO exclude transitions whose
behavior policy exceeds `max_policy_lag`; offline MARWIL does not age out
demonstrations. The fragment embedding contains mean state and state delta
plus action, reward, terminal, and length summaries.

Default utility is the robustly scaled mean replay priority:

```text
log(1 + priority)
```

An optional `qubo_reward_utility_weight` adds absolute fragment reward. It is
zero by default because reward magnitude can over-select rare noisy outcomes
and because replay priority already contains learner information. Utilities
are winsorized through 5th/95th-percentile scaling. Similarity is an RBF kernel
after median/IQR feature scaling, which reduces sensitivity to isolated
outliers.

## Configuration

```json
{
  "replay_selector": "qubo",
  "rollout_length": 32,
  "max_policy_lag": 64,
  "qubo_pool_factor": 4.0,
  "qubo_utility_weight": 1.0,
  "qubo_diversity_weight": 0.35,
  "qubo_cardinality_penalty": 4.0,
  "qubo_local_search_passes": 8,
  "qubo_reward_utility_weight": 0.0
}
```

`pool_factor` bounds QUBO construction: a batch of (k) rollout fragments
considers at most approximately `pool_factor * k` candidates. QUBO storage is
quadratic in the candidate count, so this is both a computational and memory
limit.

The latest audit record contains candidate replay indices for transition
selection, or candidate fragment keys for trajectory selection, followed by
the binary decision vector. TensorBoard and the internal metrics API record
candidate count, selected count, QUBO energy, utility, and mean selected
redundancy, plus solve and total selector time. Because incomplete terminal
fragments are not cut or padded, the actual selected transition count is also
reported and can differ from the nominal batch size.

Ordinary prioritized replay has defined sampling probabilities and uses its
importance weights. The subsequent deterministic QUBO solve changes inclusion
probabilities, so the built-in QUBO path uses neutral weights and reports
`selector_importance_correction: 0`. It does not present the candidate-pool
PER weight as an unbiased correction for the final binary decision.

## Experiment gate

The fair comparison holds constant (or explicitly normalizes for):

- actor data, random seeds, learner updates, and observed selected transitions;
- algorithm and hyperparameters;
- validation paths and checkpoint rule; and
- wall-clock accounting, including QUBO construction and solve time.

Compare QUBO against uniform replay, ordinary prioritized replay, a greedy
utility-only selector, and a non-QUBO diversity heuristic. Report held-out
return/risk, effective sample diversity, selector time, total learner time,
and sensitivity to every QUBO coefficient. A useful batch is not evidence of
quantum advantage, and the built-in solver is a classical heuristic rather
than an exact optimizer.

## Search-frontier integration

Replay selection saves learner work only after actors have produced their
experience. Search pruning moves the same yes/no allocation earlier: candidate
children receive cheap value estimates and path embeddings, QUBO retains a
fixed frontier, and only retained nodes receive expensive rollout or terminal
evaluation.

`jormungandr.search` supplies `SearchNode`, `UtilityFrontierPruner`,
`QUBOFrontierPruner`, the `jormungandr.frontier_pruners` entry-point group, and
`bounded_beam_search`. It deliberately does not define an environment. Volt
can keep graph nodes in each search node's opaque payload while exposing only
a stable key, utility estimate, and embedding. One-hot branch-position
features make shared prefixes similar; Volt can later replace them with its
Deep Sets or graph embeddings.

The expensive operation must occur after pruning. If every candidate rollout
has already been simulated to obtain its utility, QUBO cannot recover that
cost. Suitable cheap utilities include a policy/value estimate, an admissible
bound, model disagreement, or a shallow rollout.

## Controlled branch-search result

The seeded benchmark constructs a four-way tree of depth six. It has 4,096
terminal paths and 5,460 non-root candidates. Terminal values are known to the
benchmark oracle. Search sees a noisy value proxy with error correlated within
the first branch, so it tests whether diversity can hedge a systematically
optimistic region. Oracle construction is evaluation instrumentation and is
excluded from reported search accounting.

Run the paired comparison with:

```bash
PYTHONPATH=src python examples/benchmark_qubo_branch_search.py \
  --trials 500 --widths 4,6,8,10,12,16 --seed 20260802 \
  --json-output docs/latex/figures/qubo_frontier_efficiency.json \
  --plot-output docs/latex/figures/qubo_frontier_efficiency.pdf
```

Recorded on 2 August 2026:

| selector | beam | exact recovery | mean regret | p95 regret | generated | terminal evaluations | selector ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| utility | 8 | 72.2% | 0.1435 | 1.0047 | 148 | 8 | 0.09 |
| QUBO | 8 | 73.2% | 0.1305 | 0.8060 | 148 | 8 | 2.87 |
| utility | 10 | 74.4% | 0.1266 | 0.7706 | 180 | 10 | 0.11 |
| QUBO | 10 | 75.4% | 0.1135 | 0.7230 | 180 | 10 | 3.57 |

At width eight, paired mean-regret improvement was `+0.0130` in QUBO's favor
(normal 95% confidence interval `+0.0022` to `+0.0239`) and exact recovery
improved by one percentage point. QUBO width eight was close to utility width
ten: it used 32 fewer generated candidates and two fewer terminal evaluations,
at the cost of about 2.75 ms additional selection time. The terminal-only
break-even cost was therefore about 1.38 ms per avoided evaluation; savings in
intermediate expansion would lower that threshold when expansion itself is
costly.

The committed [vector efficiency figure](latex/figures/qubo_frontier_efficiency.pdf)
shows all six budgets with normal mean-regret and Wilson recovery intervals;
the adjacent JSON file contains its complete aggregate data.

## Branch-by-branch playback

The wording matters: QUBO is applied once per expanded frontier, not once
inside every branch. If `B_(t-1)` is the retained beam, all of its children are
first collected into `C_t`; one binary decision vector over `C_t` then defines
`B_t`. The committed [animated trace](markup/qubo-frontier.gif) displays that
recursion, the actual normalized utilities and similarity matrix, and every
`x_i = 0/1` decision for seed `20260802`.

Regenerate the explanatory trace and its paper figure with:

```bash
PYTHONPATH=src python examples/visualize_qubo_frontier.py \
  --seed 20260802 --beam-width 8 \
  --json-output docs/latex/figures/qubo_frontier_trace.json \
  --plot-output docs/latex/figures/qubo_frontier_trace.pdf \
  --gif-output docs/markup/qubo-frontier.gif
```

That seed is illustrative and is not used as independent statistical evidence.
The paired 500-tree result above supplies the empirical comparison.

This is evidence that the integration can improve a controlled noisy search,
not evidence of a Volt improvement, general QUBO superiority, an exact solve,
or quantum advantage. The proxy is synthetic and defined as oracle subtree
value plus correlated noise to isolate pruning behavior. The next gate is the
same paired accounting on immutable Volt path cohorts using a
production-available cheap estimate.
