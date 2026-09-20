# Questions

Statuses: `DECIDED` (user answered) · `ASSUMED` (default taken, correct it if
wrong) · `FORK` (waiting on the user) · `DEFERRED` (not needed this milestone).

## Open forks

Empty. The September 20 revision introduced no low-confidence, high-rework fork
that could not be settled by the user's stated goal or current competition
evidence. New defaults Q14–Q20 are explicit below.

## Register

| ID | Question | Status | Answer or default | Landed |
|----|----------|--------|-------------------|--------|
| Q1 | Who is this for, and are there conflicting actors? | ASSUMED | Solo repo owner; the global budget scheduler overrides an environment agent when compute is scarce | PLAN §Users and actors |
| Q2 | What's in scope for this milestone? | ASSUMED | Reproduce Duck, build the hybrid hypothesis/verifier loop, ablate it, and make September/November submissions; training is gated, not required | PLAN §Scope |
| Q3 | What's the trace/episode identity? | ASSUMED | Extend the existing JSONL schemas; address evidence by `run_id/environment_id/level/step` | ADR-0006, PLAN §Implementation decisions |
| Q4 | Where does state live? | ASSUMED | Isolated flat-file episode workspaces; no database | PLAN §Shape |
| Q5 | How are concurrent writes handled? | ASSUMED | One writer per environment workspace; global metrics append through a coordinator | PLAN §Shape |
| Q6 | What's the interface boundary? | ASSUMED | Provider-agnostic core with official-starter local/Kaggle adapters | ADR-0001 |
| Q7 | What happens on malformed frames or failed inference? | ASSUMED | Preserve normal terminal empty-frame behavior; bounded retry, then ordered degradation/fallback without aborting the submission | ADR-0001, PLAN §Implementation decisions |
| F1 | Which v1 inference runtime? | DECIDED, SUPERSEDED | The earlier llama.cpp/GGUF choice was superseded by Qwen 3.6 27B FP8/vLLM after Duck validated it on Kaggle | ADR-0009 supersedes ADR-0002 |
| F2 | Is satay-runtime included? | DECIDED | No | ADR-0008 |
| F3 | Where does GPU-dependent work happen? | DECIDED | Kaggle notebooks primary; rented RTX PRO 6000-class GPU fallback | ADR-0004 |
| Q8-model | Default model? | ASSUMED, SUPERSEDED | Qwen 3.6 27B FP8 replaces Qwen3-30B-A3B | ADR-0009 supersedes ADR-0007 |
| Q8-gen | How are synthetic games generated? | DEFERRED | Choose only if Q18's fine-tuning gate opens; no synthetic pipeline in the September critical path | ADR-0012 |
| Q8-teacher | Which frontier teacher supplies traces? | DEFERRED | Select by the specific failure cluster only after Q18's gate opens | ADR-0012 |
| Q9 | Where and within what limit does scoring run? | DECIDED | Offline Kaggle GPU notebook, official limit nine hours—not the earlier 12-hour assumption | PLAN R0; Kaggle rules |
| Q10 | What is measurable success? | ASSUMED | First: valid full run and nonzero score; candidates must improve paired level depth/RHAE without violating runtime; final: best measured variant | PLAN R0, R7 |
| Q11 | What is sensitive? | ASSUMED | Kaggle/API credentials never enter logs, traces, datasets, notebooks, or commits; scored submission needs no secret | PLAN §Scope |
| Q12 | How does stored data evolve? | ASSUMED | `schema_version` on trace records plus explicit world-model revision IDs | ADR-0006 |
| Q13 | Is test-time training in scope? | DECIDED | No per-environment or in-session weight updates | ADR-0005 |
| Q14 | Fork Duck wholesale or use it as a control? | ASSUMED | Pin/reproduce it and reuse tested components, while keeping this project's independent core and official adapter | ADR-0011 |
| Q15 | Textual facts or executable Python world models? | ASSUMED | Hybrid ledger; promote only valuable, testable claims to replay-verified code | ADR-0010 |
| Q16 | Image, raw grid, or objects as the observation? | ASSUMED | Current rendered image plus compact symbolic segmentation/delta and lossless query tools | ADR-0010 |
| Q17 | What does the scheduler optimize? | ASSUMED | Level completion/depth first, then action efficiency, subject to an 8h30m internal deadline | PLAN §Implementation decisions |
| Q18 | Is offline fine-tuning required? | ASSUMED | No; open the gate only for a common, transferable measured failure and admit by held-out paired evaluation | ADR-0012 |
| Q19 | Does learned game knowledge cross environment boundaries? | ASSUMED | No; persist across levels within one environment and reset at the next environment | ADR-0010 |
| Q20 | What evaluation-set size drives budgeting? | ASSUMED | Budget and simulate for 110 isolated environments until the exact official end-to-end lifecycle proves otherwise | PLAN §Assumed defaults |

## Coverage

| Category | Covered by |
|----------|-----------|
| Primary user and actors | Q1 |
| Scope boundary | Q2, Q13, Q18 |
| Data model and identity | Q3, Q12 |
| State and storage | Q4, Q19 |
| Concurrency and conflict | Q5, Q20 |
| Interfaces and contracts | Q6, Q14, Q15 |
| Failure behaviour | Q7, Q17 |
| External dependencies | F1, F2, Q8-model, Q8-gen, Q8-teacher, Q14 |
| Runtime and deployment | F3, Q9, Q17, Q20 |
| Measurable success | Q10, Q17, Q18 |
| Security and secrets | Q11 |
| Versioning and migration | Q12 |
| Perception and state grounding | Q16 |
| Learning and generalization boundary | Q13, Q18, Q19 |
