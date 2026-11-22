# Changes

## 0.3.0

This release adds optional language policies, multi-step credit assignment, and
ranked-shortlist feedback primitives. The revised paper describes their tests,
including auxiliary objectives that did not materially improve learning.

- Optional LoRA language-policy plugins and verified tool-routing examples.
- Structured DQN n-step returns with episode, discount, and causal-history checks.
- Generic auxiliary-loss composition; no auxiliary objective enabled by default.
- Ranked exposures, selections, ratings, ties, explicit preference losses, and
  an in-memory feedback join. No automatic preference-to-reward conversion.
- Replicated inference, indexed replay, optional native episode learners, SBE
  transport for fixed-vector inference, and verified optional ONNX exports.
- Controlled benchmarks and source/result provenance for the research additions.

Structured experience now requires an explicit reward, as the protocol already
specified. An omitted reward is rejected rather than converted to zero. Empty
entity observations now retain their feature width through JSON serialization.
For aggregated returns, the construction discount must match the learner.

Preference ingestion remains a Python library, not a durable HTTP service.
The numeric studies are not additional LLM fine-tuning results, and their
simulation assumptions are not framework defaults.
