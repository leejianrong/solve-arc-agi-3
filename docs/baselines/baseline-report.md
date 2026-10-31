# Baseline report (ARC-41)

Everything measured about the pinned control and our agent up to 2026-10-31. Details and raw numbers live in
[`kaggle-submission-checklist.md`](kaggle-submission-checklist.md) and the evidence folders; this page is the summary
and the way to reproduce each number.

## The control

Qwen3.6-27B-FP8 (`vrfai/Qwen3.6-27B-FP8`, revision `0766367`) served by vLLM 0.19.0 from the pinned offline wheelhouse
(see [`duck-control-v1.md`](duck-control-v1.md)). Our own agent drives it through an OpenAI-compatible endpoint; the
agent is a clean-room single-step tool-calling loop, not Duck's REPL harness (ADR-0011).

## What the hardware does (Kaggle RTX PRO 6000, offline)

| Measurement | Value | Source |
|---|---|---|
| Offline venv build | 21 to 62 s warm, 22 minutes once on a cold mount | serve probe, dry runs |
| Full model hash check | about 175 s | serve probe |
| Weights load | 5 min 22 s to 8 min | serve probe |
| Engine start-up after load (compile, graph capture) | about 207 s | vLLM log |
| Model ready after run start | 467 to 689 s in the dry runs (a cold run took about 35 minutes) | `run_summary.json` |
| Decode throughput | 39 tok/s for one stream, 270 aggregate at 8, 780 aggregate at 32 (24 per stream) | serve probe |
| KV cache | 178k tokens (43.6 GiB), 10x concurrency at 65k | vLLM log |
| RunPod, same GPU class | 38.7 tok/s single stream, 351 s model load | [`evidence/runpod-acceptance-2026-09-20/`](evidence/runpod-acceptance-2026-09-20/) |

The Kaggle image needs two fixes the RunPod run did not: `/usr/local/nvidia/lib64` on `LIBRARY_PATH` (FlashInfer's JIT
link failed on `-lcuda`) and a Python 3.12 venv, because the default is 3.13 and the wheelhouse is cp312.

## What the agent did, run by run

| Run | Agent | Settings | Result |
|---|---|---|---|
| Dry run 1, 6 games | first agent | thinking on, 4096 tokens, 120 s | 70% of steps fell back (timeouts, no action in the reply); 0 levels |
| Dry run 3, 12 games | bounded context | thinking off, 1024 tokens, 180 s | 23% fallbacks, 0 levels, 292 tokens and 30 s per call |
| Submission 1 (hidden set) | same as run 3 plus scanning fallback clicks | thinking off | **score 0.00**, ref 57788190 |
| Dry run 5, 12 games | v5 (objects, diffs, plans, memory) | thinking off | 1,928 actions, 11 fallbacks (0.6%), 1 level (`lp85` level 1), score 0.0346, 124 tokens and 11.5 s per call |
| Dev run, 8 games | v5 | thinking ON, 6144 tokens, 900 s | 608 actions, 0 levels, 38.6 s per call: thinking does not pay here |
| Random-plan floor, 24 games | scripted, no model | 150 to 200 steps, 6 seeds | mean 0.067 (0.0 to 0.163), nearly all luck on level 1 of `sp80`, `vc33`, `ft09` |

Reading it: the pipeline is sound and the model now makes valid moves, but the agent plays at about the level of
random exploration on the public set. The score is 100 x (baseline/actions)^2 per level, weighted by level number, so
what is missing is clearing levels in close to human-baseline action counts, not speed.

## Problems found and fixed on the way

A parser that rejected valid replies (extra `index` field on OpenRouter tool calls; `action(...)` written in prose;
clicks with no coordinates), an unbounded prompt that would have overflowed after about ten steps, counters read as game
state, a wheelhouse dataset that was deleted and re-created with a different lock, doubled log lines that inflated
action counts, and a silent no-op edit that wasted a one-hour run.

## Reproduce

```sh
make check                                        # 184 tests, strict types
PYTHONPATH=src ../play-arc-agi-3/.venv/bin/python scripts/local_play.py --fake --games 24 --steps 150 --trace-dir /tmp/t
scripts/kaggle_dev.sh <name> ARC3_LEDGER=1        # one variant on Kaggle, summary from recordings
python3 scripts/summarize_run.py <run folders>    # compare runs
```
