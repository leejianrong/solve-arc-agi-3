# solve-arc-agi-3: Plan

Status: agreed · Milestone: ARC Prize 2026 Milestone 2 and final submission

## Problem

ARC-AGI-3 asks one agent to enter an unfamiliar interactive 64x64 world,
discover its goal and mechanics, and complete progressively harder levels with
few actions. The Kaggle submission must do this offline in a GPU notebook that
finishes within nine hours. The final evaluation contains hidden environments,
so memorizing the 25 public demonstration games is useless at best and harmful
at worst.

Frontier systems have now nearly saturated ARC-AGI-3 when paired with good
context management or long-horizon harnesses, but those results use models and
test-time compute that cannot ship in Kaggle. The relevant open-weight result is
Tufa Labs' Milestone 1-winning Duck harness: Qwen 3.6 27B FP8, vLLM, a Python
REPL, compact perception tools, and short rolling context. It proves a 27B
coding model can run inside the competition envelope, but its reported public
demonstration score was only 1.60 RHAE. The problem is therefore not whether a
27B model loads; it is how to spend its limited reasoning tokens and environment
actions on persistent, falsifiable understanding.

## Solution

Build a budget-aware coding-agent harness around Qwen 3.6 27B. It records every
transition losslessly, maintains a compact ledger of competing hypotheses, and
uses surprise to decide when another model call is worth its cost. Most rules
remain cheap structured or textual claims. When a rule has enough evidence and
would affect several future actions, the model promotes it into a small Python
world model, patched through a fixed interface and accepted only if it replays
the observed history exactly.

The agent plans against verified code, executes plans one real action at a time,
and interrupts immediately when reality differs from prediction. It retains
knowledge across levels of the same environment, but starts each new environment
with a clean workspace. A dynamic scheduler enforces the notebook's global
deadline and degrades gracefully from verified planning to direct action rather
than timing out.

## Users and actors

The primary user is the repo owner, developing and submitting a personal Kaggle
entry. Runtime actors are one isolated agent workspace per environment, a shared
local vLLM server, the ARC game engine, and the global budget scheduler. The
scheduler wins whenever an agent asks for more compute than the submission can
afford.

## Scope

**In this milestone.**

- Reproduce the open-source Duck/Qwen 3.6 27B baseline on the actual Kaggle RTX
  PRO 6000 environment before building on it.
- Package one offline submission through the official ARC-AGI-3 Kaggle starter.
- Preserve lossless frames and transitions while presenting the model with a
  compact current image, symbolic delta, segmentation, and query tools.
- Maintain evidence-linked hypotheses with confidence, predictions, supporting
  transitions, and counterexamples.
- Support a gated executable world model with incremental patches, sandboxed
  execution, exact replay verification, and rollback.
- Search the verified model for short plans and execute them with prediction
  checks after every real action.
- Enforce global wall-clock, per-environment, context, model-call, and output-token
  budgets, with measured degradation modes.
- Run controlled ablations and submit by September 30 and November 2, 2026.

**Out.**

- Test-time weight updates or per-environment LoRA training (ADR-0005).
- Game-specific code, prompts, public-game manuals, or memory shared between
  environments. They would optimize the demonstration set rather than hidden
  transfer.
- A second game UI. Existing replay and trace viewers are sufficient.
- A knowledge-graph service or database. Episode-local JSON/JSONL and Python
  files are inspectable, cheap, and enough.
- Synthetic-game generation and QLoRA as roadmap dependencies. They are allowed
  only after a measured harness failure shows what data would fix and a small
  pilot passes the gate in ADR-0012.
- Multi-model debate, fixed multi-agent roles, and unrestricted tree search.
  Their token and latency cost is a poor fit for the nine-hour notebook.

## Requirements

| ID | Requirement | Status |
|----|-------------|--------|
| R0 | Produce a valid offline Kaggle submission using a 20–30B open-weight model that completes within the official nine-hour GPU limit | Core goal |
| R1 | Reproduce a pinned Duck/Qwen baseline and measure RHAE, throughput, VRAM, latency, and projected full-run time on Kaggle hardware | Must-have |
| R2 | Preserve lossless episode evidence while keeping the active model context bounded and useful across long games | Must-have |
| R3 | Track falsifiable hypotheses and revise or retire them from observed counterexamples | Must-have |
| R4 | Promote selected hypotheses into incrementally patchable Python and accept them only after deterministic replay verification | Must-have |
| R5 | Plan with verified rules, execute with per-action prediction checks, and fall back safely when the model is wrong or unavailable | Must-have |
| R6 | Stay below configurable whole-run, per-environment, call, token, and action budgets without losing the whole submission to one game | Must-have |
| R7 | Compare direct REPL, ledger-only, and ledger-plus-executable variants on identical seeds/runs before choosing what ships | Must-have |
| R8 | Admit fine-tuning only if a failure-derived pilot improves held-out transfer enough to repay its runtime and maintenance cost | Deferred gate |

## Shape

| Part | Mechanism | ADR |
|------|-----------|-----|
| S1 | Official starter-kit deployment shell with the existing provider-agnostic core and thin local/Kaggle adapters | ADR-0001 |
| S2 | One local vLLM server hosting Qwen 3.6 27B FP8 on the Kaggle RTX PRO 6000 | ADR-0009 |
| S3 | Multi-resolution observation cache: rendered current frame, palette/connected components, object deltas, frame hashes, and on-demand crop/grid queries | ADR-0010 |
| S4 | Episode workspace containing an append-only transition log, current task/plan, hypothesis ledger, and accepted world-model revisions | ADR-0010 |
| S5 | Event-driven model loop invoked on level start, prediction failure, exhausted plan, or high-value uncertainty—not automatically after every action | ADR-0010 |
| S6 | Fixed world-model contract (`parse`, `transition`, `render`, `terminal`, `legal_actions`) edited by constrained patch operations in a sandbox | ADR-0010 |
| S7 | Replay verifier plus bounded BFS/A*/beam planner; plans abort on the first observed mismatch | ADR-0010 |
| S8 | Global deadline scheduler allocating calls/tokens across active environments and switching through explicit degradation modes | ADR-0009 |
| S9 | Versioned traces and paired evaluation reports, extending the existing recording schemas | ADR-0006, ADR-0011 |

## Affordances

**UI.** No new UI. Runs are inspected through generated score reports and the
existing `play-arc-agi-3`/Duck viewers.

**Non-UI.**

| Affordance | Kind | Wires to |
|------------|------|----------|
| `solve-arc-agi-3 benchmark duck` | CLI command/notebook entry | Pinned Duck control, S2, S9 |
| `solve-arc-agi-3 run --variant=<direct\|ledger\|verified>` | CLI command | S1–S9 local evaluation |
| `solve-arc-agi-3 replay-verify <run>` | CLI command | S4, S6, S7 |
| `solve-arc-agi-3 compare <baseline> <candidate>` | CLI command | Paired per-game metrics and runtime gates |
| Official `MyAgent.choose_action` / `is_done` | Kaggle entry point | S1, S5, S8 |
| `runs/<run_id>/` | Append-only flat-file store | Full frames, transitions, hypotheses, code revisions, usage, outcomes |

## Implementation decisions

Qwen 3.6 27B FP8 through vLLM replaces Qwen3-30B-A3B through
GGUF/Ollama/llama.cpp (ADR-0009). This is not a paper comparison: the Duck
harness has already validated the new stack in the same competition, while the
old notebook never produced a completed baseline. FP8 is the quality-first
default on the 96GB Blackwell GPU. NVFP4/AWQ is a measured fallback only if FP8
throughput, KV cache, or concurrency is the actual bottleneck.

Duck is the control and a source of proven designs, not an invisible upstream
dependency (ADR-0011). Its package metadata declares an MIT classifier, but no
root license file was found at the expected published path during this review.
Copy code only after a concrete license grant is verified; otherwise reimplement
the documented segmentation, Python-tool, packaging, simulator, and evaluation
behavior. Keep this project's core contract and official starter adapter so every
change is independently testable.

Each environment owns an isolated workspace. A transition is addressed by
`run_id/environment_id/level/step`; every hypothesis cites transition IDs. A
hypothesis has a stable ID, claim, scope, confidence band, predicted observable
effects, evidence, counterexamples, and status (`candidate`, `accepted`,
`rejected`, `superseded`, `executable`). Confidence is never model prose alone:
the harness changes status from deterministic checks and explicit model actions.

The executable model is a small deliverable behind a fixed contract, not an
unbounded game rewrite. The model emits a unified diff or structured replacement
for one allowed file. The harness parses the AST, rejects forbidden imports and
IO, compiles in a fresh process, runs resource limits, and exact-replays all
applicable recorded transitions. A failed patch is retained as evidence but never
becomes active. A passing patch can still be partial; its declared scope controls
which transitions and plans may rely on it.

Context has three tiers: the current observation and compact ledger in the prompt;
lossless frames/transitions behind query tools; accepted executable knowledge on
disk. Deterministic compaction keeps goals, accepted rules, unresolved
contradictions, and the active plan. Raw old frames are evicted from the prompt,
not from storage. This applies the strongest consistent lesson across OpenAI's
retained-reasoning result, VISTA, AVO, Tycho, and Duck: long-horizon state must
survive individual model turns.

The hard runtime is nine hours. The initial operating envelope is an 8h30m
internal deadline: at most 15 minutes for startup/model warm-up, 7h20m for game
work, 25 minutes for orchestration/output, and 30 minutes of failure reserve.
The scheduler derives per-environment allowances from remaining wall time,
active concurrency, and unfinished environments after V1 measures the real
throughput. Initial per-environment guards are 12 model turns, 4,096 generated
tokens, and one code-promotion attempt per level; these are hypotheses to tune,
not promises. Degradation is ordered: shorten output → stop code promotion → use
ledger-only action selection → deterministic exploration fallback → stop the
environment cleanly before the global deadline.

Fine-tuning is not the first lever (ADR-0012). Public demonstration traces are
evaluation/debug data, not training targets. If the base model repeatedly fails a
specific transferable skill—tool syntax, concise hypothesis updates, patch repair,
or counterexample use—build a small synthetic or teacher-labeled dataset for that
skill, train an adapter offline, and admit it only if paired held-out tests improve
RHAE or level depth without breaking the runtime budget.

External dependency licenses are part of the run manifest. The official
Qwen 3.6 27B weights and vLLM are Apache-2.0. Any community FP8/NVFP4 checkpoint
must preserve the model license and publish its quantization provenance. Duck
source reuse remains blocked until its distribution license is verified; the
official starter is used as the competition-prescribed interface and its
redistribution terms must be checked before vendoring it into a public release.

## Testing approach

The highest-value tests are behavioral. Scripted hidden-rule games test whether
the ledger changes after counterexamples and whether code promotion reproduces
every prior transition. Recorded real episodes test deterministic replay,
compaction, and plan interruption without spending environment actions. A
110-environment clone run tests packaging, isolation, concurrency, and the global
deadline. Candidate harnesses use identical model settings and repeated public
runs; decisions use paired per-game level depth/RHAE plus runtime, not a single
aggregate or cherry-picked best run.

## Assumed defaults

| ID | Assumed | Cost if wrong |
|----|---------|---------------|
| Q14 | Use Duck as the pinned control and component reference, but keep an independent core | Medium: a wholesale fork would accelerate packaging but couple experiments to TAAF internals |
| Q15 | Hybrid textual/structured hypotheses with gated executable promotion | High: making either text-only or code-only the universal representation could waste the remaining competition window |
| Q16 | Give the VLM both the current rendered frame and compact symbolic tools | Medium: dropping either representation may reduce perception or consume excess context |
| Q17 | Optimize first for level completion/depth, then action efficiency, while enforcing the global runtime | Medium: RHAE rewards both, but incomplete later levels cap the score sharply |
| Q18 | Defer fine-tuning until a failure-derived pilot passes a held-out gate | Medium: an excellent existing adapter could be left unused; the harness remains swappable |
| Q19 | Persist knowledge across levels only, never across environments | Low: cross-environment memory can be added later, but risks contamination and overfitting now |
| Q20 | Plan against 110 isolated evaluation environments and a 9-hour run until an official end-to-end test proves a different orchestration shape | Medium: budget formulas change, not the agent contracts |

## Open risks

- Qwen 3.6 27B may be too weak to repair a persistent simulator reliably even
  though it can use a transient Python REPL. V2/V3 compare ledger-only and verified
  code directly; executable promotion is removable, not load-bearing.
- The official Kaggle runner's concurrency and lifecycle may differ from the local
  110-clone simulator. V1 packages and executes the exact starter path before any
  architecture work.
- Public-game saturation makes public RHAE a weak predictor of hidden transfer.
  Evaluation therefore emphasizes fresh scripted games, run isolation, repeated
  trials, and the Kaggle leaderboard rather than optimizing individual public
  failures.
- Exact pixel replay can reject useful abstract rules because animation or hidden
  state is omitted. The contract permits scoped/object-level invariants, but every
  relaxation must be explicit and separately measured in V3.
- Continuous batching may improve throughput while harming time-to-first-token and
  memory at high concurrency. V1 measures the concurrency frontier on the actual
  RTX PRO 6000 instead of copying Duck's settings blindly.
- Ten days remain before Milestone 2 as of September 20. V4 ships the best proven
  variant even if the executable branch has not earned inclusion.

## Evidence behind the revision

- [Kaggle competition page](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview): nine-hour GPU limit, offline execution, September 30 and November 2 deadlines.
- [Official Kaggle starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter): starter contract, accelerator choices, and RTX `g4-standard-48` deployment path.
- [Duck harness](https://github.com/Tufalabs/duck-harness) and [Tufa Labs report](https://tufalabs.ai/research/duck-harness/): Milestone 1-winning Qwen 3.6 27B FP8/vLLM REPL baseline.
- [ARC-AGI-3 technical report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf): RHAE, hidden-set design, action budget, and overfitting cautions.
- [Executable world-model study](https://arxiv.org/abs/2605.05138) and [ablation](https://arxiv.org/abs/2607.15439): verification is promising but expensive; persistent code is not uniformly better than text.
- [Tycho](https://github.com/NIMI-research/Tycho/), [VISTA](https://vista-research.github.io/), [NVIDIA AVO](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/), and [OpenAI's context study](https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/): complementary evidence for persistent evidence, falsification, lossless observation access, and context compaction.
