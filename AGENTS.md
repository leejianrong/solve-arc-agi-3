# AGENTS.md

## WHY

A custom agent harness for ARC-AGI-3 (arcprize.org) that must run offline on a Kaggle GPU with an
open-weight ~27B model (Qwen3.6-27B-FP8 on vLLM). `docs/PLAN.md` is the agreed design (hypothesis
ledger, executable world model, scheduler), `docs/SLICES.md` says what is built, and the code is what
runs today: a much simpler single-step tool-calling agent. Trust the code over any prose, and fix the prose.

Sibling repos: `../play-arc-agi-3` (human play app, local games in `environment_files/`, its `.venv` has
`arc_agi`), `../arc-agi-3-benchmarking`, `../../arc-agi-1` (unrelated older benchmark).

## Hard rules (each cost real time or money more than once)

1. **A scripted edit can silently do nothing.** `ruff format` reflows code, so exact-string replaces stop
   matching. Assert the match count in the patch script and re-grep afterwards. A no-op edit once wasted a
   one-hour Kaggle run.
2. **Money.** Never use OpenRouter (any model). Give the user a cost estimate and get an OK before using any
   paid key or RunPod. Never touch a pod, key or limit you did not create. RunPod jobs follow the
   `runpod-jobs` skill; the dead-man's-switch is armed first. Launch with `scripts/runpod_launch_lab.sh up`, then
   always `watch --max-usd N` (pod-side `lab_obs.sh` + driver `runpod_watch.py`: phase deadlines, fatal lines, evidence shipped before delete).
   RunPod pods on driver 570 cannot run our pinned vLLM wheels (PTX error); prefer Kaggle dev kernels.
3. **Count from recordings or the trace, not log lines.** Kaggle logs print every agent line twice.
4. **Submissions.** One per day, deadline 2026-11-02 23:59. Never submit without the user's go-ahead. Facts,
   pins and the dry-run history: `docs/baselines/kaggle-submission-checklist.md`.
5. **A Kaggle dry run does not test the submission path.** Dry runs use `dry_runner.py` and offline games; a
   rerun uses the official starter and the gateway, and both our submissions scored 0.00 with no logs to read.
   Kaggle also mounts inputs in two layouts (nested under `competitions/`, or flat): never hard-code a mount path.
6. **The score rewards few actions, not speed.** Level score = 100 x (baseline/actions)^2, level i is weighted
   i. Wasted moves on level 1 cost points; later levels matter most.
7. Never commit `.env`, keys or recordings. The Kaggle source dataset holds code only.

## Commands

```sh
make check                 # lint + mypy + tests; run before calling anything done
PYTHONPATH=src ../play-arc-agi-3/.venv/bin/python scripts/local_play.py --fake --games 24 --steps 150
                           # free scripted client through the real agent and scorer (crashes, format bugs)
scripts/build_kaggle_bundle.sh "msg"   # publish code to the private dataset (BUNDLE_REF=<commit> ships an older one)
scripts/kaggle_dev.sh <name> KEY=VAL   # dev run on its own dataset, so a submission's code stays put
kaggle kernels push -p kaggle/<dir>    # dev kernels: kaggle/dev-think; submission: kaggle/submission
```

Model behaviour tests use **semula + Haiku subagents** only: `docs/SEMULA-HAIKU-LOOP.md`. Real-Qwen
checks run as Kaggle dev kernels (free). Never call a hosted model API. Every Haiku study ends with matplotlib
charts and a written report, not only a table. One run clears 0 to 3 levels, so repeat before comparing variants.

## Layout

`src/solve_arc_agi_3/`: `agent.py` (prompt, parsing, plans, effect memory), `observation.py` (text view of a frame,
motion, counters), `kaggle_agent.py` (competition wrapper), `step_trace.py` (per-step trace, `docs/TRACE.md`; the
older `trace.py` is only the smoke test's event log), `ledger.py` (hypothesis ledger), `world_model.py` (contract,
replay check) and `sandbox.py` (runs model-written code; not wired in yet). `kaggle/run/` is what runs in the
notebook (`main.py`), `scripts/` holds tooling (`local_play.py`, RunPod lab), `tools/replay/index.html` is the trace
viewer (`make e2e-viewer` tests it in a real browser). Switches: `ARC3_TRACE_DIR` (write traces), `ARC3_LEDGER=1`
(ledger block in the prompt), `ARC3_PROMPT_CHARS=n` (prompt budget), `ARC3_EVENTS=1`, `ARC3_TOOLS=1`, `ARC3_WORLD_MODEL=1` (needs tools), `ARC3_NAV=1` (state graph, reachability and loop warning in the STATUS header), `ARC3_CLICK_HINTS=auto|wins|sweep|none` (click guidance in the header; `auto` is today's behaviour); all off by default.

## Git

Work on a `feat/` branch off fresh `main`, one slice per branch, and land it through a PR (no direct pushes to
`main`; CI runs `make check`). Put the net line change in the PR description. Commit with the noreply identity through `-c user.name=... -c user.email=85485699+leejianrong@users.noreply.github.com`
(do not change git config). Push and open PRs only when asked.

## Subagents

At most **2 running at once**, and a subagent must **not spawn subagents**. Give each one a disjoint set of
files and say which. Subagents never commit, push, submit, spend money, or run repo-wide formatters
(`ruff format` on a path you do not own rewrites someone else's work). They run `make check` only at the end,
report what they changed, and the main agent reviews, integrates and commits.
