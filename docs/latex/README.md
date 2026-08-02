# Jörmungandr paper

[Jörmungandr: A Plugin Runtime for Distributed Reinforcement Learning](jormungandr_learning_systems.pdf)
([source](jormungandr_learning_systems.tex)) describes the actor/learner
boundary, algorithm and selector plugins, checkpoint lifecycle, noise and
return-risk taxonomy, QUBO rollout allocation, Volt set/graph integration,
the controlled QUBO frontier-search result, a seeded online-agent convergence
comparison, and the staged Rust boundary.

The paper follows the plain 11-point article format used by the public Volt
and Hypercube papers: Palatino and Helvetica typefaces, compact margins, and a
standard title and abstract without a separate cover or contents page. It
distinguishes the implemented fixed-discrete-action service and synthetic
search benchmark from graph-native service transport, reference DreamerV3
parity, native kernels, and Volt-scale empirical learning claims that remain
future work.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  jormungandr_learning_systems.tex
```

`latexmk -c` removes intermediate files without deleting the PDF or source.

Regenerate the Section 6 frontier trace, empirical data, and vector figures
from the repository root with:

```bash
PYTHONPATH=src python examples/visualize_qubo_frontier.py \
  --json-output docs/latex/figures/qubo_frontier_trace.json \
  --plot-output docs/latex/figures/qubo_frontier_trace.pdf \
  --gif-output docs/markup/qubo-frontier.gif
PYTHONPATH=src python examples/benchmark_qubo_branch_search.py \
  --trials 500 --widths 4,6,8,10,12,16 --seed 20260802 \
  --json-output docs/latex/figures/qubo_frontier_efficiency.json \
  --plot-output docs/latex/figures/qubo_frontier_efficiency.pdf
```

Regenerate the Section 8 online-agent comparison and Markdown playbacks with:

```bash
PYTHONPATH=src python examples/compare_ou_algorithms.py \
  --json-output docs/latex/figures/ou_algorithm_convergence.json \
  --plot-output docs/latex/figures/ou_algorithm_convergence.pdf \
  --convergence-gif-output docs/markup/algorithm-convergence.gif \
  --playback-gif-output docs/markup/algorithm-playback.gif
```

Plotting uses the optional `plot` dependency group; GIF rendering additionally
uses `record`.
