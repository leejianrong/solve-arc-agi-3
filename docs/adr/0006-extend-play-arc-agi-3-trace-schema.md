# ADR-0006: Extend play-arc-agi-3's trace schema instead of inventing a new one

- Status: Accepted
- Date: 2026-08-31
- Deciders: repo owner

## Context

The fine-tuning pipeline (collect, generate, filter, train) needs a trace
format that records enough of a game episode to be useful as SFT data:
frames, actions, outcomes, and eventually the harness's own detected objects
and discovered facts. `play-arc-agi-3` (a sibling repo) already has a
versioned JSONL recording format, one line per step, schema-versioned, with a
lean mode that skips frames since the games are deterministic and frames can
be reconstructed by replay. It exists for manual-play session recording, not
for training-data collection, but it already captures frame, action, and
state per step.

## Decision

The training-trace format is `play-arc-agi-3`'s existing schema plus new
object and relation fields (detected objects from state grounding, discovered
facts from mechanism discovery), versioned forward under the same
`schema_version` convention rather than as a second, parallel format.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Design a new trace schema from scratch, purpose-built for SFT | Duplicates a format that already exists and already has a versioning story, for no benefit beyond a cleaner slate; two formats for the same underlying data (a game episode) is a maintenance cost with no offsetting upside |
| Use a published agent-trajectory schema unmodified (e.g. Agent Data Protocol's `Trajectory{Action, Observation}`) | Reasonable shape, but doesn't natively carry ARC-AGI-3's frame/level/game structure the way `play-arc-agi-3`'s schema already does; better as a reference for what fields to add than as the format itself |

## Consequences

This buys one format across manual-play recordings and training traces, so a
session recorded for one purpose is usable for the other without translation,
and it buys a versioning story (`schema_version`) that already exists rather
than one that has to be invented. It costs a dependency on `play-arc-agi-3`'s
schema staying stable or being extended carefully. Adding the object/relation
fields has to happen without breaking existing recordings, which is exactly
what `schema_version` is for, so this is a constraint to respect, not a risk
to work around.
