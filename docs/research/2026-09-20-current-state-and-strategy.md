# ARC-AGI-3 current state and competition strategy

Date: 2026-09-20

This memo records the evidence behind the project plan in [`../PLAN.md`](../PLAN.md) and the delivery slices in [`../SLICES.md`](../SLICES.md). It replaces the August research snapshot as the current planning reference.

## Executive finding

A competition-scale system should not try to imitate the frontier systems by spending more tokens on every step. The strongest practical starting point is a capable 27B model, an offline inference server, and a harness that preserves evidence across turns. The harness should let the model form falsifiable hypotheses, test them through environment interaction, and promote only verified rules into executable code.

The recommended first model is Qwen3.6 27B FP8 served with vLLM. The recommended control is a clean reimplementation of the public Duck architecture after its distribution licence is verified. The first research question is whether an evidence-linked hypothesis ledger improves that control. Executable Python world models come after that and must pass a three-way ablation against direct action and ledger-only reasoning.

This approach fits the current Kaggle shape: one offline RTX PRO 6000 class GPU, a nine-hour run, and a submission artifact that must include all weights and dependencies.

## What the headline results do and do not show

OpenAI reported 99.95% on the semi-private ARC-AGI-3 evaluation with GPT-6 Astra. NVIDIA reported 100% on the public set with its Adaptive Verifier Orchestrator. These results establish that the benchmark is tractable for strong interactive agents. They do not establish that a single 20B to 30B model can reproduce those scores inside a Kaggle notebook.

The reported systems benefit from frontier reasoning, large inference budgets, orchestration, or evaluation conditions that differ from the competition's hidden test. NVIDIA's result covers the public set of 25 games and 183 levels, so it should not be treated as evidence of private-set generalisation. OpenAI's result is a more demanding semi-private evaluation, but its model and serving stack are outside the intended competition budget.

The useful lesson is architectural. Both results point toward persistent reasoning, verification, and adaptive control rather than one-shot visual question answering.

## Best available competition-scale baseline

The most relevant public baseline is Tufa Labs' Duck, the Milestone 1 winner. Its published system uses Qwen3.6 27B FP8 with vLLM, a Python REPL, image segmentation, and a compact interaction loop. Its reported public demonstration mean was 1.6002.

Duck is valuable as a control because it already exercises the model, server, visual tools, and environment loop at roughly the target scale. It should not become an application framework that the project must preserve. We need a pinned, reproducible control and clear extension seams for memory, verification, and planning.

Before copying upstream source, verify the concrete licence distributed with the exact revision. The repository metadata advertises MIT, but the expected root licence file was not available during this review. If that cannot be resolved, reimplement the documented behaviour without copying source.

## Evidence about executable world models

Recent ARC-AGI-3 work on executable world models supports the user's core idea, with an important qualification. A model can maintain Python code that predicts transitions, replay observations against that code, and plan through the resulting simulator. Complete verification can improve reliability, but persistent executable code is not uniformly beneficial. Wrong code can become a durable false belief, and exhaustive checking consumes actions and wall time.

The resulting design should use two representations:

1. A structured hypothesis ledger for facts, unknowns, competing explanations, supporting observations, contradictions, confidence, and the next discriminating experiment.
2. A small executable model only for hypotheses that have enough transition evidence and can be checked cheaply.

Executable rules should never be accepted because the model sounds confident. Promotion requires replay against stored transitions. A failed rule should be patched through a bounded interface or rolled back to its last verified revision.

## Proposed agent loop

The loop separates evidence collection from commitment:

1. Parse the observation into a lossless transition record plus compact visual summaries.
2. Update the hypothesis ledger with evidence links and explicit contradictions.
3. Choose whether the next step is information gathering, progress, recovery, or exploitation.
4. If a rule is mature enough, emit a bounded patch to the executable world model.
5. Validate the patch, replay all relevant transitions, and reject or roll back inconsistent revisions.
6. Search a small action horizon in verified code when that is cheaper than another model call.
7. Execute one environment action and record the result.
8. Compact only when an event threshold is reached, preserving raw transition data outside the prompt.

The LLM should not rewrite an unrestricted game implementation on every turn. It should operate on a stable schema and a constrained patch protocol. This reduces token use, makes failures attributable, and prevents accidental damage to the harness.

## State and prompt efficiency

The raw state belongs in deterministic storage, not in the conversational transcript. Each transition should retain the action, before and after grids, score or reward changes, termination flags, extracted objects, and any tool output needed for replay.

The model receives a task-dependent view:

- A compact scene graph or object table for routine decisions.
- A local crop or difference map when a change is spatially limited.
- The full grid only when topology, colour, or global layout matters.
- A ledger summary containing unresolved contradictions and the highest-value tests.

Model calls should be event-driven. Useful triggers include a newly observed mechanic, a contradiction, a failed prediction, a milestone, a recovery state, or exhaustion of the current plan. Deterministic code should handle repeated movement, bookkeeping, replay, and short-horizon execution.

## Compute plan

The scheduler should treat nine hours as a hard envelope and reserve time for startup and final packaging. A provisional budget is:

| Activity | Budget |
| --- | ---: |
| Notebook startup, model load, and warm-up | 35 minutes |
| Calibration and early-game exploration | 55 minutes |
| Main inference and interaction | 390 minutes |
| Verification, recovery, and adaptive reserve | 40 minutes |
| Output validation and shutdown margin | 20 minutes |

These are initial controls, not constants. The first GPU benchmark must measure model load time, prompt throughput, generation throughput, peak memory, and stable concurrency on the actual Kaggle hardware. The scheduler can then allocate token and action budgets dynamically by remaining time and observed progress.

FP8 is the default because the target hardware supports it and the strongest relevant baseline already uses it. More aggressive quantisation should be tested only if it creates useful headroom without degrading visual reasoning, code generation, or long-context consistency.

## Fine-tuning decision

Fine-tuning is not part of the initial critical path. The first gains are more likely to come from evidence management, verification, and model-call scheduling. A generic ARC fine-tune also risks teaching benchmark style without improving interactive control.

Open a QLoRA or adapter-training experiment only when all of the following are true:

- Repeated traces show a stable skill deficit rather than a harness defect.
- The target behaviour can be scored automatically on held-out games.
- The training set is procedurally generated or otherwise free of hidden-test leakage.
- The adapter still fits the offline artifact and runtime budget.
- A paired evaluation can isolate the adapter's contribution.

Likely candidates are structured state extraction, choosing discriminating experiments, and repairing a world model from a counterexample. End-to-end game imitation is a lower-priority candidate because it is harder to validate and more likely to overfit.

## Evaluation gates

Every architectural addition must beat a simpler control under matched budgets.

| Gate | Compared systems | Promotion rule |
| --- | --- | --- |
| Baseline | Pinned Duck-style control versus direct model loop | Keep the faster system unless completion or depth improves materially |
| Ledger | Direct action versus evidence-linked ledger | Promote only with higher completion or depth at acceptable action cost |
| World model | Direct action versus ledger-only versus verified executable model | Promote only if replay verification improves score or recovery enough to repay its overhead |
| Quantisation | FP8 versus smaller formats | Promote only if quality remains within the predeclared tolerance and runtime headroom is useful |
| Fine-tuning | Base model versus adapter | Promote only on held-out interactive traces with the same harness and budget |

The primary metrics should follow the competition incentives: level completion, depth reached, total score, and action efficiency. Model calls, generated tokens, verifier failures, rollbacks, wall time, and peak memory are diagnostic metrics.

## Main risks

- **False executable beliefs:** a plausible but wrong simulator can waste many actions. Mitigate with replay checks, confidence thresholds, bounded patches, and rollback.
- **Verification overhead:** full replay can consume the budget. Use dependency-aware checks and reserve complete verification for high-impact changes.
- **Prompt bloat:** a growing narrative memory will eventually dominate inference. Keep raw evidence in deterministic storage and render compact task-specific views.
- **Public-set overfitting:** tuning to known games can produce impressive local results without transfer. Use procedural perturbations and hold out entire mechanic families.
- **Kaggle packaging failure:** a strong local agent is irrelevant if its model or dependencies cannot load offline. Build and rehearse the submission artifact early.
- **Unclear baseline provenance:** copying code without a verified licence creates avoidable risk. Resolve provenance before implementation.

## Immediate experiments

The implementation order is intentionally conservative:

1. Reproduce the 27B FP8 baseline and measure the true resource envelope.
2. Add lossless transition storage, deterministic visual summaries, and the evidence ledger.
3. Compare the ledger against the baseline on matched seeds and budgets.
4. Add a constrained executable rule format, sandbox, replay verifier, revision history, and short-horizon planner.
5. Run the three-way ablation and keep executable modelling only if it earns its cost.
6. Submit an early competition artifact, analyse scored failures, and spend the remaining cycle on the highest-value transferable mechanism.

The corresponding acceptance criteria and target dates are in [`../SLICES.md`](../SLICES.md).

## Sources

- [ARC Prize: OpenAI GPT-6 Astra results](https://arcprize.org/results/openai-gpt-6-astra)
- [NVIDIA: AVO reaches 100% on ARC-AGI-3](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/)
- [Tufa Labs: Duck harness report](https://tufalabs.ai/research/duck-harness/)
- [Tufa Labs Duck harness repository](https://github.com/Tufalabs/duck-harness)
- [Executable World Models for ARC-AGI-3](https://arxiv.org/abs/2605.05138)
- [Executable world-model ablation](https://arxiv.org/abs/2607.15439)
- [ARC Prize: ARC-AGI-3 competition](https://arcprize.org/arc-agi/3/)
- [Kaggle: ARC Prize 2026, ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
- [Google Cloud: RTX PRO 6000 GPU documentation](https://cloud.google.com/compute/docs/gpus/rtx-pro-6000)
- [Qwen model repository and licence](https://huggingface.co/Qwen)
- [vLLM documentation](https://docs.vllm.ai/)

Source claims are snapshots as of the memo date. Competition rules, available Kaggle hardware, model releases, and public leaderboards should be rechecked before each submission milestone.
