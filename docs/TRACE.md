# Trace format (schema version 1)

The agent writes one trace per game. It is the lossless transition log that `PLAN.md` asks for first
(card ARC-43): the replay viewer (`tools/replay/`) reads it, the hypothesis ledger takes its evidence
from it, and a world model is accepted only if it replays it exactly.

The official Kaggle recordings are not enough: `action_input` is always 0 with no data, and they hold no
model reply, plan or timing.

## Files

```text
traces/<run_id>/
  run.json              one object: settings, model, bundle commit, limits, games, scorecard (filled at the end)
  <game_id>.jsonl       one JSON object per line, in order, written as the game is played
  <game_id>.obs.jsonl   optional (switch ARC3_TRACE_OBS=1): {"step": n, "text": "..."} the text the model saw
```

Default location: `$ARC3_TRACE_DIR/<run_id>` (unset means tracing is off). `run_id` defaults to a UTC timestamp.
Writers append one line at a time and flush, so a killed run leaves a readable trace.

## Records

Every line has `v` (always 1) and `kind`.

### `kind: "start"` (first line of a game)

| field | meaning |
|---|---|
| `t` | seconds since the run started |
| `game` | game id, e.g. `lp85-305b61c3` |
| `state`, `levels`, `win_levels`, `avail` | as the engine reported before any move; `avail` is a list of action numbers |
| `frame` | `{"base": <rows>}`, the full first frame (encoding below) |

### `kind: "step"` (one per move played)

| field | meaning |
|---|---|
| `t`, `game` | as above |
| `step` | 0-based index of the move in this game |
| `src` | `model` (move from a model call), `queued` (rest of an earlier plan), `planned` (move from the planner over a verified world model, `ARC3_WORLD_MODEL=1`; carries `plan` and `plan_idx` like `queued`; additive, schema version stays 1), `fallback` (scripted), `forced_reset` |
| `moves` | the move played as a one-item list: `[{"name": "ACTION6", "x": 12, "y": 30}]` (`x`, `y` only for clicks) |
| `state`, `levels`, `win_levels`, `avail` | as reported after the move |
| `call` | only when `src` is `model`: `{"ms": 11500, "prompt_tokens": 5100, "completion_tokens": 124, "ok": true}` |
| `reply` | only on a model call: `{"content": "...", "reasoning_tail": "...", "tool_calls": ["{\"code\": ...}"]}`; content cut to 2000 characters, reasoning to its last 300 |
| `plan`, `plan_idx` | on a model call and on its queued moves: the whole list of moves the model asked for, and which one this is |
| `nudge` | true when this call carried the plain "reply with a tool call" reminder |
| `event` | optional string, only with the `ARC3_EVENTS=1` switch and only on a model call: why the call happened, one of `level_start`, `surprise` (a queued move contradicted a supported ledger claim and the rest of the plan was dropped), `no_progress` (two moves in a row changed nothing), `plan_exhausted`. Additive; the schema version stays 1 |
| `error` | only on a fallback: why the model step failed (exception class and message, cut to 300 characters) |
| `change` | `{"n": 293, "bbox": [x0, y0, x1, y1], "regions": [[x0, y0, x1, y1, cells], ...]}`; `n` is 0 and `bbox` and `regions` are omitted when nothing changed (at most 6 regions, largest first) |
| `motion` | objects that kept colour and size but moved: `[{"color": 5, "size": 10, "from": [x, y], "to": [x, y]}]` (centres); empty list if none |
| `counters` | regions flagged as a probable counter or timer at this point, as the strings the model was shown |
| `effects` | per-action memory as the model saw it: `{"ACTION1": [tries, tries_that_changed_the_board]}` |
| `frame` | the board after the move, one of `{"diff": [[y, x0, x1, c], ...]}` (changed runs against the previous frame, `c` a colour 0 to 15) or `{"key": <rows>}` (used when the diff has more than 150 runs) |

A move that ends the game still gets a step record. Fields that do not apply are omitted, never null.

## Frame encoding

A frame is 64 rows of 64 hex digits (colour 0 to 15 as `0`-`f`), encoded as a list of strings, one per
distinct run of identical rows:

- A row is run-length encoded: tokens joined by `.`, each token is the colour character followed by the run length,
  so `51.363` is colour `5` once, then colour `3` sixty-three times.
- A row identical to the previous one is not repeated: `~n` after a row string means "and n more identical rows".

- `~n` is a **suffix on the row string** (`08~5`), not a separate list item.
- A diff run `[y, x0, x1, c]` covers columns `x0` to `x1` **inclusive** and sets them to colour `c`.
- A diff is against the **previous record's frame** in the same file: the `start` base for the first step, the
  previous step's frame afterwards. A keyframe replaces the running frame outright.
- Frames are 64 rows of 64 digits in practice; readers need not enforce it.
- `counters` strings never contain numbers that change between steps (they describe rows, columns or boxes).

This is the encoding the design mock-up used; its round trip was checked against real frames.

## Example (one step, abridged)

```json
{"v":1,"kind":"step","t":412.8,"game":"lp85-305b61c3","step":43,"src":"model","moves":[{"name":"ACTION6","x":12,"y":30}],
 "state":"NOT_FINISHED","levels":1,"win_levels":8,"avail":[6],
 "call":{"ms":11500,"prompt_tokens":5100,"completion_tokens":124,"ok":true},
 "reply":{"content":"Clicking the left button...","reasoning_tail":"","tool_calls":["{\"code\": \"action([{'action': 6, 'x': 12, 'y': 30}])\"}"]},
 "plan":[{"name":"ACTION6","x":12,"y":30}],"plan_idx":0,"nudge":false,
 "change":{"n":1524,"bbox":[0,0,63,51],"regions":[[0,0,63,51,1524]]},"motion":[],"counters":[],
 "effects":{"ACTION6":[43,28]},"frame":{"key":["e1.363","..."]}}
```

## Compatibility

Readers must ignore unknown fields and unknown `kind` values. A change that removes or renames a field bumps `v`.
