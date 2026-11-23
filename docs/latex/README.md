# Jörmungandr papers

Two papers cover distinct applications of the framework:

- [A Plugin Runtime for Distributed Reinforcement Learning](jormungandr_learning_systems.pdf)
  ([source](jormungandr_learning_systems.tex)): the original 15-page paper on RL
  algorithms, infrastructure, structured actions, search, and financial
  application interfaces. Source and PDF are restored from revision `07ce89c`,
  before the language-model material was incorporated.
- [Adapting a Language Model to Tool Use](jormungandr_llm_adaptation.pdf)
  ([source](jormungandr_llm_adaptation.tex)): the language-policy experiment,
  comparisons with simple rules, and separate numeric studies of context
  objectives, delayed rewards, and preferences. Those numeric results are not
  presented as improvements already demonstrated in an LLM.

The journal class uses a compact two-column A4 layout with full-width figures.
The companion hedging paper develops the stock-and-option application further.
No models were retrained and no measurements changed for this split.

Regenerate the two feedback-study vector figures from committed results, without
training models, from the repository root:

```bash
python examples/plot_feedback_paper.py
```

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  jormungandr_learning_systems.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  jormungandr_llm_adaptation.tex
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

Regenerate the independent CartPole PPO reference comparison from the pinned
SB3 CPU environment with:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV PYTHONPATH="$PWD/src" \
  /path/to/sb3-reference/bin/python examples/benchmark_gym_ppo.py \
  --runs 3 --total-timesteps 49152 --rollout-steps 1024 \
  --evaluation-every-timesteps 4096 --evaluation-episodes 20 \
  --json-output docs/latex/figures/cartpole_ppo_reference.json \
  --plot-output docs/latex/figures/cartpole_ppo_reference.pdf
```

Regenerate the entity/candidate CartPole representation parity control with:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV PYTHONPATH="$PWD/src:$PWD" \
  /path/to/sb3-reference/bin/python \
  examples/benchmark_structured_cartpole_ppo.py \
  --flat-reference docs/latex/figures/cartpole_ppo_reference.json \
  --json-output docs/latex/figures/structured_cartpole_parity.json \
  --plot-output docs/latex/figures/structured_cartpole_parity.pdf
```

Regenerate the four-way masked Taxi constraint control with:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV PYTHONPATH="$PWD/src" \
  /path/to/sb3-reference/bin/python \
  examples/benchmark_masked_taxi_ppo.py \
  --runs 3 --total-timesteps 131072 --rollout-steps 2048 \
  --evaluation-every-timesteps 16384 --evaluation-episodes 50 \
  --json-output docs/latex/figures/masked_taxi_ppo_reference.json \
  --plot-output docs/latex/figures/masked_taxi_ppo_reference.pdf
```

Regenerate the generic constrained joint-action J1 gate with:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV PYTHONPATH="$PWD/src:$PWD" \
  /path/to/sb3-reference/bin/python \
  examples/benchmark_constrained_workbench.py \
  --runs 3 --ppo-total-turns 12288 \
  --bc-updates 500 --bc-train-episodes 256 \
  --bc-validation-episodes 96 --evaluation-episodes 64 \
  --json-output docs/latex/figures/constrained_workbench_j1.json \
  --plot-output docs/latex/figures/constrained_workbench_j1.pdf
```

Regenerate the isolated prefix-preference gate with:

```bash
python examples/benchmark_prefix_conditioning.py
```

Regenerate the conditional supervision-sampling gate with:

```bash
PYTHONPATH=src python examples/benchmark_conditional_supervision_sampling.py \
  --json-output docs/latex/figures/conditional_supervision_sampling.json
```

Regenerate the 719-step terminal-credit reference gate with:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV PYTHONPATH="$PWD/src:$PWD" \
  /path/to/sb3-reference/bin/python \
  examples/benchmark_delayed_terminal_credit.py \
  --json-output docs/latex/figures/delayed_terminal_credit.json
```

Regenerate the structured PPO critic-isolation and transactional-ratio gate
with:

```bash
PYTHONPATH=src python examples/benchmark_structured_ppo_safety.py \
  --json-output docs/latex/figures/structured_ppo_safety.json
```

Plotting uses the optional `plot` dependency group; GIF rendering additionally
uses `record`.

## Stock-and-option hedging tutorial

[Learning to Hedge with Stocks and Options](hedging/volt_jormungandr_option_hedging.pdf)
explains the financial control problem and native learner API. Its
[canonical source and build instructions](hedging/README.md) live in the
`hedging/` subdirectory, with a scoped journal template.
