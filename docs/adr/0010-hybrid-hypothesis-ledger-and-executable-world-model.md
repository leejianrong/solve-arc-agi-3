# ADR-0010: Hybrid hypothesis ledger with gated executable world models

- Status: Accepted
- Date: 2026-09-20
- Deciders: repo owner (assumed default from the proposed approach and evidence)

## Context

The user's proposed loop is hypothesis formation, Python implementation,
falsification against the live game, revision, and finally planning through the
learned rules. Frontier-agent results support that shape, but the July ablation
of executable world models found two important limits: persistent code was not
uniformly better than a textual model, and exact verification performed best
while using substantially more resources. A 27B model has less coding and repair
capacity than the frontier models in those studies.

The original plan reacted by excluding executable models and keeping only a
structured fact list. That gives up the main benefits of code—exact replay,
counterexamples, and cheap simulated planning—precisely where a weaker model
needs deterministic help.

## Decision

Use a hybrid representation.

Every environment starts with a compact, evidence-linked hypothesis ledger.
Claims remain textual/structured while evidence is sparse or their expected
future reuse is low. Promote a claim to executable Python only when it has a
testable prediction, enough supporting transitions, and expected planning value.

Executable models implement a fixed, narrow interface: `parse`, `transition`,
`render`, `terminal`, and `legal_actions`. The model edits one file through
bounded patches. The harness statically validates, sandboxes, compiles, exact-
replays applicable history, and rolls back failures. Accepted code may declare a
partial scope. Plans generated from it execute one real action at a time and stop
on the first prediction mismatch.

The model loop is event-driven. Level starts, unexpected transitions, exhausted
plans, and high-value uncertainty can trigger inference; routine predicted steps
do not.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Text/structured facts only | Cheap, but cannot be replay-verified or searched reliably and repeats reasoning at action time |
| Executable model from the first transition | Forces premature precision and expensive repair on a small model before enough evidence exists |
| Free-form Python REPL only, as in Duck | Proven baseline and retained as the control, but transient snippets do not create a verified cumulative simulator |
| Separate actor and world-model agents | Frontier systems can afford it; duplicating long contexts and model calls is poorly matched to a nine-hour local 27B run |

## Consequences

The executable branch can fail without sinking the agent: ledger-only direct
action remains a complete fallback and an ablation control. The cost is a more
complex state machine, explicit scope tracking, and a sandbox/verifier. Incremental
patching and exact replay become core test targets rather than prompt suggestions.
