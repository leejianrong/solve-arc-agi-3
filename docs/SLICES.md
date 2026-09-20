# solve-arc-agi-3: Slices

Vertical increments. Each ends in something demonstrable. V1 confronts the
riskiest immediate unknown: whether the only proven 27B competition stack can be
reproduced, packaged, and projected under the current nine-hour limit.

## V1: Reproduce the Duck/Qwen competition baseline

**Target:** September 20–22 · **Delivers:** R0 (partial), R1, R6 (measurement)

**Implementation note (2026-09-20):** the CPU-only artifact slice now includes
typed provider/runtime configuration, offline preflight, an owned-process vLLM
lifecycle, the official `choose_action`/`is_done` adapter, run-scoped traces, and
a deterministic no-network fake-server smoke. OpenRouter currently lists
`qwen/qwen3.6-27b`, but it is explicitly development-only. Full mounted-asset
preflight and exact RunPod FP8 smoke/performance evidence remain required before
V1 or ARC-42 can be considered complete.

**Build plan**

1. Pin a Duck commit, its Qwen 3.6 27B FP8 weights, vLLM wheelhouse, prompts,
   runtime flags, and available license metadata; do not modify the control or
   copy its code into the submission until a concrete license grant is verified.
2. Run its public-game smoke test on the Kaggle RTX PRO 6000 and record startup,
   VRAM, time to first token, prefill/decode rate, model-call latency, RHAE, level
   depth, actions, and failures.
3. Sweep only the variables that determine feasibility: 1/4/8/16 concurrent
   contexts and the smallest context lengths that preserve behavior. Run an FP8
   versus NVFP4/AWQ quality/throughput pilot only if FP8 misses the projection.
4. Package the control through the official starter and run Duck's 110-clone
   competition simulator under an 8h30m watchdog.
5. Replace the obsolete GGUF notebook with a reproducible benchmark notebook and
   emit a machine-readable baseline report.

**Demo:** One command/notebook produces a pinned Duck score report and a runtime
projection that says `fits`, `does not fit`, or `unknown` with the missing datum.

**Rests on assumptions:** Q14 (Duck is the control) and Q20 (budget for 110
environments). If Q20 is wrong, scheduler constants change; the measurements and
agent boundary do not.

### Test plan

#### End-to-end

- The exact offline Kaggle artifact starts vLLM, completes a representative game
  batch, writes traces/submission output, and exits before its watchdog.
- The 110-clone simulator completes with isolated workspaces and no leaked game IDs,
  conversations, or files between clones.

#### Integration

- The official `MyAgent` adapter returns valid simple and coordinate actions from
  Duck's local OpenAI-compatible server, including terminal-frame handling.
- The report reconciles model usage, actions, outcomes, and elapsed wall time from
  one real run.

#### Unit

- Deadline projection and degradation thresholds are correct at boundary values.
- Upstream/model/wheel hashes and hardware metadata are present in every report.

## V2: Evidence memory and falsifiable hypotheses

**Target:** September 23–25 · **Delivers:** R2, R3, R6 (first policy)

**Build plan**

1. Adapt the proven segmentation/tool surface so every transition stores full
   frames, hashes, action, object/delta summaries, level status, and model usage.
2. Add an episode-local hypothesis ledger with stable IDs, predictions, evidence,
   counterexamples, scope, and status; expose bounded query/update tools to the
   model instead of asking it to restate the whole history.
3. Add deterministic compaction that preserves the current goal, accepted rules,
   open contradictions, active plan, and tool results while evicting raw history
   only from the prompt.
4. Trigger model calls at events (level start, surprise, plan exhaustion,
   high-value uncertainty) and let predicted routine steps execute without a new
   inference call.
5. Compare `direct` (pinned Duck) and `ledger` on scripted unseen games and a
   repeated subset of public games with identical model/runtime settings.

**Demo:** A replay shows a wrong hypothesis receiving a counterexample and being
revised, followed by several correctly predicted actions executed without
per-action model calls; the report compares score, calls, tokens, and time.

**Rests on assumptions:** Q16 (image plus symbolic tools), Q17 (completion/depth
before efficiency), and Q19 (no cross-environment memory).

### Test plan

#### End-to-end

- On a hidden-rule fixture game, the agent records an initial wrong rule, rejects
  or supersedes it after contradictory evidence, and completes the level.
- On the paired evaluation subset, the ledger variant stays within budget and
  reports its delta from Duck without cherry-picking runs.

#### Integration

- Context compaction preserves accepted hypotheses and unresolved contradictions
  after raw messages exceed the configured window.
- Loading a recorded transition log reconstructs the same ledger and next prompt.

#### Unit

- Frame hashing, segmentation, delta extraction, hypothesis state transitions,
  and evidence references are deterministic on fixtures.
- No record from environment A is queryable from environment B.

## V3: Gated executable hypotheses and planning

**Target:** September 26–28 · **Delivers:** R4, R5, R7 (partial)

**Build plan**

1. Define the small world-model contract: `parse`, `transition`, `render`,
   `terminal`, and `legal_actions`, with declared partial scope.
2. Add bounded patch tools for one model file, AST/import validation, subprocess
   resource limits, revision history, and atomic rollback.
3. Build the verifier: exact replay by default, with separately reported scoped
   object-level checks only where full rendering is intentionally unsupported.
4. Promote code only when a hypothesis has a falsifiable prediction, supporting
   evidence, and expected reuse; cap attempts by the scheduler.
5. Add bounded BFS/A*/beam planning over verified transitions. Execute plans one
   real action at a time and abort immediately on a prediction mismatch.
6. Run the same paired evaluation as V2 across `direct`, `ledger`, and `verified`.

**Demo:** The model patches a wrong transition function, the first revision fails
replay and rolls back, the second passes, and the planner uses it to complete a
fixture level; the paired report shows whether the feature earned its cost.

**Rests on assumptions:** Q15 (hybrid promotion). If small-model code repair is
unreliable, the feature flag remains off and V2 is still submission-ready.

### Test plan

#### End-to-end

- A fixture requiring a revised rule cannot be solved by the initial model, then
  completes after a replay-verified patch and planned action sequence.
- The verified variant either passes the paired acceptance gate or is excluded
  from V4 with the decision recorded in the report.

#### Integration

- Every accepted revision replays all in-scope transitions; adding a contradictory
  transition invalidates the revision and triggers replanning.
- A running plan is interrupted before its second wrong real action when the first
  predicted successor differs from observation.

#### Unit

- Forbidden imports, filesystem/network access, syntax errors, timeouts, and
  oversized patches are rejected without changing the active revision.
- Planner limits on nodes, depth, and time are deterministic and enforced.

## V4: Milestone 2 submission

**Target:** September 29–30 · **Delivers:** R0, R6, R7 (first Kaggle result)

**Build plan**

1. Select the highest paired-performing V1–V3 variant that passes the runtime
   gate; do not hold the submission for an unproven feature.
2. Freeze hashes, budgets, seeds, dependencies, model data, license notices, and
   the full degradation policy.
3. Run the 110-clone preflight and the official starter's local verification.
4. Submit Phase A, inspect artifacts/errors, then spend one official Phase B
   submission and record the leaderboard score and kernel runtime.
5. Triage the scored run by failure class, not by public game identity.

**Demo:** A completed Kaggle kernel and Milestone 2 score with a reproducible run
manifest and no internet dependency.

**Rests on assumptions:** none new.

### Test plan

#### End-to-end

- Kaggle Phase A and Phase B both complete under nine hours and produce a valid
  competition score.

#### Integration

- The packaged notebook resolves every import/model artifact with internet off and
  exits cleanly when the internal 8h30m deadline is forced in a shortened test.

#### Unit

- Manifest validation fails before upload on missing hashes, model data,
  dependencies, or license metadata.

## V5: Failure-driven improvement gate

**Target:** October 1–18 · **Delivers:** R7, and R8 only if its gate opens

**Build plan**

1. Cluster V2–V4 failures into perception, goal inference, exploration, memory,
   tool syntax, code repair, planning, and runtime starvation using traces rather
   than anecdotes.
2. Implement the cheapest harness correction for the largest transferable cluster
   and rerun the paired evaluation.
3. If and only if a common cluster is clearly trainable, create a small synthetic
   or teacher-labeled skill dataset without public-game solutions, run QLoRA, and
   compare the adapter against the unchanged base model.
4. Reject any change that improves a few public games while regressing unseen
   fixture transfer, full-run projection, or run isolation.

**Demo:** A ranked failure report and one accepted improvement with paired evidence;
if the fine-tuning gate stays closed, the report states why and no training stack
is built.

**Rests on assumptions:** Q18 (fine-tuning must earn entry).

### Test plan

#### End-to-end

- The accepted candidate improves paired level depth/RHAE with no runtime-budget
  failure; otherwise the unchanged V4 artifact remains the champion.

#### Integration

- Any adapter loads through the same vLLM/agent interface and can be disabled by
  configuration without code changes.

#### Unit

- Failure labels link to concrete transitions and sum to the evaluated failures.
- Dataset validation rejects public game IDs, solution traces, malformed tool
  calls, and examples without provenance.

## V6: Final ablation and submission

**Target:** October 19–November 2 · **Delivers:** R0, R7, R8 decision

**Build plan**

1. Freeze direct, ledger, verified, and any admitted adapter variants.
2. Run repeated paired evaluation with identical budgets and report mean/variance,
   per-game depth, RHAE, calls, tokens, actions, and wall time.
3. Choose the champion by the prespecified gate: completes the full projection,
   maximizes paired level depth, then RHAE, with no unexplained catastrophic
   regressions.
4. Run offline packaging/preflight and submit before the November 2 deadline.

**Demo:** The final ablation table, champion manifest, and final Kaggle score.

**Rests on assumptions:** none new; this slice resolves Q18 with evidence.

### Test plan

#### End-to-end

- Every finalist completes the same evaluation protocol, and the selected artifact
  completes Kaggle Phase A/Phase B within the hard limit.

#### Integration

- Re-running champion selection over the frozen reports returns the submitted
  manifest hash.

#### Unit

- Aggregation, paired deltas, confidence intervals, and tie-breaking reproduce
  expected results on fixture reports.
