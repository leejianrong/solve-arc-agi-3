# Testing the agent with semula and Haiku subagents

A model-in-the-loop check with no hosted API: the agent calls semula, an OpenAI-compatible server that
parks every unknown prompt until a coding agent answers it. Haiku subagents play the part of the model.
It tests whether prompts and observations are understandable, not how Qwen behaves, and 50 calls per game
is too few to measure level clears.

## Start the server

```sh
cd ~/projects/abang-ai/semula
curl -s 127.0.0.1:8080/health || SEMULA_MODE=record docker compose -p semula-arc3 up -d --wait semula
```

Dashboard: http://127.0.0.1:8080/ui/ (every prompt and answer). Stop it with
`docker compose -p semula-arc3 down`. Semula replays identical prompts from its store, so an old wrong answer
replays too; use a fresh project or volume for a clean run.

## Run the game side

```sh
set -a; . ./.env; set +a   # not needed for semula, only for other tools
PYTHONPATH=src ../play-arc-agi-3/.venv/bin/python scripts/local_play.py \
  --games ls20,ft09,cd82 --steps 60 --config configs/baselines/semula-haiku.dev.json --timeout 600 --concurrency 3
```

## The players

Spawn one Haiku subagent per game (`model: haiku`, background, within the 2-subagent cap, so run two at a
time and the third after one ends). Shard by the `Available actions:` header, which sits in the first 200
characters of each call (`semula pending` truncates `last_message` there). Each player loops, over Bash only:

```sh
docker exec semula-arc3-semula-1 semula pending --wait 60 --json
docker exec semula-arc3-semula-1 semula show <ID> --json
docker exec -i semula-arc3-semula-1 semula answer <ID> --stdin <<'EOF'
{"message": {"content": null, "tool_calls": [{"function": {"name": "python", "arguments": {"code": "action([1, 4])"}}}]}}
EOF
```

Rules for the players: use only the request contents (no repo or game source), answer only their game, stop
after 4 empty waits or 50 answers, then report specifically what they could not tell from the text.

## What it cost and gave

Three players answered about 135 calls in about 5 minutes at 95 to 105k tokens each. The reports are claims:
check them against our own logs. They found real bugs in our observation text each time.

## Benchmarking a variant

Goal: compare a variant (say `ARC3_LEDGER=1`) with a baseline on the same games, N Haiku players per variant,
and read the result from our own traces. Steps for the main agent. Check `local_play.py --help` first: a
switch only works if the code reads it. Today `local_play.py` has `--ledger` and `--trace-dir` as flags;
`ARC3_EVENTS`, `ARC3_TOOLS`, `ARC3_WORLD_MODEL` (needs tools) and `ARC3_PROMPT_CHARS` are read from the
environment by `agent_switches`; `ARC3_LEDGER` is read only by the Kaggle entry, so use `--ledger` here.
`local_play.py` does not write `.obs.jsonl` files (`keep_obs` is off), so the traces hold steps only.
A variant that needs a new flag (for example `--nav`) must be wired in `local_play.py` first.

1. **Pick the plan.** A label per variant (`base`, `ledger`), a fixed game list (3 to 4 games, for example
   `ls20,ft09,cd82`), `--steps 50` (the player cap is 50 answers per game), R repeats per label. Every run
   is one player per game, so N players per run = number of games. State the number of Haiku subagents
   and their expected tokens (about 100k each, from "What it cost") to the user before spawning.
2. **Fresh semula state per run.** Semula replays identical prompts, so an old answer would leak into the next
   run, and parked calls from earlier runs would be mixed up with this one. Use one project per run. The
   `default` project already holds stale entries and pending calls, so never use it for a benchmark:

   ```sh
   RUN=base-r1   # slug: lowercase letters, digits, hyphens
   docker exec semula-arc3-semula-1 semula project create $RUN
   docker exec semula-arc3-semula-1 semula pending --project $RUN --json   # must print []
   ```

   The project is chosen by the URL: `http://127.0.0.1:8080/v1/projects/$RUN/chat/completions`. Copy
   `configs/baselines/semula-haiku.dev.json` to `build/bench/$RUN.json` with `base_url` set to
   `http://127.0.0.1:8080/v1/projects/$RUN` (the client appends `/chat/completions`). Do not edit the committed config.
   Check the server first: `curl -s 127.0.0.1:8080/health` (start it as in "Start the server" if it is down).
3. **Bring up the game side**, in the background, one process for all games of the run, with the variant's
   flags and env vars (blank env for the baseline). The timeout must outlast the player (600 is
   enough; semula holds the call until answered):

   ```sh
   mkdir -p build/bench
   ARC3_EVENTS=1 PYTHONPATH=src ../play-arc-agi-3/.venv/bin/python scripts/local_play.py \
     --games ls20,ft09,cd82 --steps 50 --config build/bench/$RUN.json --timeout 600 --concurrency 3 \
     --ledger --trace-dir build/bench/traces/$RUN > build/bench/$RUN.log 2>&1   # run_in_background
   ```

   Flags for a variant go here and nowhere else (`--ledger`, `ARC3_*` env vars). Keep everything else identical
   between labels. Traces land in `build/bench/traces/$RUN/<run_id>/`.
4. **Find the calls.** Each game's calls park under the run's project. Shard by the game's action header
   (first 200 characters of `last_message`: `Available actions: ...`), because `semula pending` has no game field.
   Wait until all games have parked their first call (`semula pending --project $RUN --wait 60 --json`) and note
   each game's header and the `Game state ... levels completed 0 of N` line. If two games share a header,
   tell each player the extra distinguishing text (the `of N` level count, board size).
5. **Spawn the players, at most 2 at a time** (`model: haiku`, background). The third waits until one ends.
   A player's brief contains: the project slug (`--project $RUN` on every `pending`, `show` and `answer`), the
   action header that identifies its game, the four commands from "The players", and the rules from there
   (request contents only, no repo or game source, only its game, stop after 4 empty waits or 50 answers,
   report what it could not tell from the text). Players must not spawn subagents. The main agent does not
   answer parked calls itself.
6. **Watch progress without answering.** `docker exec semula-arc3-semula-1 semula pending --project $RUN --json | wc -l`,
   `tail build/bench/$RUN.log`, and `ls build/bench/traces/$RUN/*/`. A call parked for more than a few minutes
   with its player ended means the player stopped early: note it and start the next player for that header.
7. **End of run.** When `local_play.py` exits (it prints `TRACE_REPLAY exact`), check the trace step counts
   match the number of answers the players report (Count from traces, not logs). Check nothing is still parked, then
   `docker exec semula-arc3-semula-1 semula project delete $RUN` only if the entries are not wanted.
   Repeat steps 2 to 6 for each repeat and each label, in alternating order (`base-r1`, `ledger-r1`, `base-r2`, ...) so
   drift in the players affects both labels alike.
8. **Compare.** Same label = repeats; the cell shows `mean (min-max)`:

   ```sh
   PYTHONPATH=src ../play-arc-agi-3/.venv/bin/python scripts/bench_summary.py \
     base=build/bench/traces/base-r1 base=build/bench/traces/base-r2 \
     ledger=build/bench/traces/ledger-r1 ledger=build/bench/traces/ledger-r2
   ```

   Columns: steps, levels cleared, step of first clear, no-change share, distinct clicks, longest no-change repeat of
   one move, share of steps with `src` fallback, override or probe, and the `failure_clusters` verdict. A label
   with `ALL` lines at the bottom gives the totals. With 50 calls per game and Haiku, differences in levels
   cleared are noise; read no-change share, repeats and clusters, plus the players' reports, and say so
   in the write-up.

Cleanup: no containers are started by this procedure beyond the existing semula; delete `build/bench/` copies of
configs when finished (traces are not committed).
