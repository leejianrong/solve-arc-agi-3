# Kaggle submission checklist (ARC-106)

Findings from 2026-10-30, three days before the 2026-11-02 23:59 final deadline. Checked with the `kaggle` CLI and
public notebooks only; nothing here has been run. Items marked **to check** are unverified.

## What a submission is

- A notebook run as a **competition rerun** (`KAGGLE_IS_COMPETITION_RERUN` set), internet off. The official starter
  (`inversion/arc3-sample-submission-random-agent`) waits for a game gateway at `http://gateway:8001`, copies
  `ARC-AGI-3-Agents` from the competition input, installs `arc-agi` from the competition's `arc_agi_3_wheels`, writes a
  `.env` pointing at the gateway, and runs `python main.py --agent myagent`. A non-rerun run writes a dummy
  `submission.parquet`. Our `choose_action`/`is_done` adapter matches that `Agent` interface.
- Leaderboard on 2026-10-30: top 75.56, Tufa Labs 70.02. Our account: **no submissions**.

## Public assets (no upload needed, to check: pinned hashes)

| Need | Public Kaggle asset | Status |
|---|---|---|
| Qwen3.6 27B FP8 | `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` (29 GB, model.safetensors 35.9 GB) | Same source as `docs/baselines/duck-control-v1.md`; **verify revision 0766367 and file hashes** |
| vLLM wheelhouse | `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3` | Byte-identical to our pin (checked). The `driessmit1` original was deleted and re-created on 2026-10-05 with a different lock and fails preflight |
| Duck harness source | `jeroencottaar/taaf-kaggle-source` (dataset licence MIT; Tufa notebook says its top submission is open source) | The upstream repo still has no LICENSE file (duck-control-v1.md); **decide whether to attach this dataset** |

So the ~41 GB uploads planned in ARC-107/108 may be unnecessary. Mounting public datasets avoids them.

## How the strong entries run models (research)

- The Milestone 2 public solution (`dfranzen/arc-agi-3-milestone-2-solution`) uses **SGLang**, installed from a wheelhouse
  into a venv, serving a 4-bit (W4A16 AutoRound) Qwen3.8 model plus a speculative-decoding draft model. "transformers" in
  the notebook is the model format and offline flags, not the generation loop. It is on GPU `NvidiaRtxPro6000`.
- The Duck harness (our control) uses **vLLM** behind an OpenAI-compatible API.
- No public entry generates with plain `transformers` `generate()`: it is too slow for thousands of agent calls in nine
  hours. All use a local OpenAI-compatible inference server.

## Open
- Daily submission limit and Phase A/B rules (**to check**: competition rules page, not readable by CLI).
- A from-`/kaggle/input` run of our notebook on the real GPU (ARC-110).

## Probe results (2026-10-29/30, private kernel `leejianrong/arc3-env-probe`, offline, `NvidiaRtxPro6000`)

Source: `kaggle/probe/`. The run took under two minutes.

- GPU NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB; driver 580.178 (CUDA 13.0 capable); 46 CPUs; 176 GB RAM.
- **Default Python is 3.13**, but `/usr/bin/python3.12`, `uv` (/usr/local/bin) and `ninja` exist. The pinned wheelhouse is
  cp312, so create the venv with `uv venv --python /usr/bin/python3.12`.
- **`nvcc` 12.8 is installed** (`/usr/local/cuda`, matches the pinned torch cu128) with `curand_kernel.h`, plus gcc/g++.
  The RunPod JIT failures (no compiler, no toolkit) do not apply here.
- Disk: `/kaggle/working` is 20 GB (too small for the model); `/tmp` has about 1 TB free. Use `/tmp` for the venv.
- Inputs mount at `/kaggle/input/competitions/arc-prize-2026-arc-agi-3` and `/kaggle/input/datasets/<owner>/<slug>`.
- ARC-107/108 (uploads) are unnecessary: the pinned model and wheelhouse are the public datasets, and they mount.

## Wheelhouse drift (found by the first serve probe, 2026-10-30)

The first run of `kaggle/run/serve_probe.py` failed preflight: `SHA256SUMS` was 20,949 bytes (pinned 21,285) and the
lock hash differed. The README of `driessmit1/arc3-vllm-h100-wheelhouse-v3` says it was re-created on 2026-10-05 after
deletion. The mirror `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3` (2026-09-20) matches the pins, so the manifest now
points there. Both datasets are third-party and could be deleted before the deadline; a private mirror of each is a
cheap insurance step (ARC-111). The same probe showed: offline venv build 21 s, full model hash check about 175 s, both
passing (model files verified).

## Dry runs on Kaggle (offline, public games from the competition mount, `NvidiaRtxPro6000`)

Notebook `leejianrong/arc3-duck-qwen`, source dataset `leejianrong/solve-arc-agi-3-src`. Games start at T+4 s on the system
Python while the venv and vLLM boot in a background thread (model ready at T+544 to 689 s; the venv took 25 to 62 s except
once, 22 minutes, on a cold mount).

| Run | Settings | Steps | Fallback steps | Levels | Notes |
|---|---|---|---|---|---|
| 1 (6 games) | thinking on, 4096 tokens, 120 s | 506 | 354 (190 timeouts, 164 no action) | 0 | thinking answers need more than 4096 tokens at about 25 tok/s per stream |
| 2 (12 games) | invalid: the tunables silently did not apply | 642 | 484 | 0 | same failure as run 1; fixed, test added |
| 3 (12 games) | thinking off, 1024 tokens, 180 s | 1602 | 364 (no valid action) | 0 | 0% replies without a tool call, mean 292 tokens and 30 s per call |

Throughput measured by the serve probe: 39 tok/s for one stream, 270 aggregate at 8, 780 at 32 (24 per stream); KV cache 178k tokens.

What the clean-room agent still cannot do: it repeats `action(1)`, click-only games (`ft09`, `lp85`) need ACTION6 coordinates it
rarely supplies, and its history is only a changed-cell count. It has not cleared a level in any run. A rerun through the
real gateway has not been exercised yet; only a real submission does that.

## Submission 1 (2026-10-30)

- Kaggle submission ref **57788190**, from `leejianrong/arc3-duck-qwen` version 4, bundle commit `c16d400`.
- Settings: Qwen3.6 27B FP8 + vLLM 0.19.0, clean-room single-step agent, thinking off, 1024 output tokens, 180 s timeout,
  32 concurrent model calls, deadline 8h30m after start. The commit run (non-rerun) took 1534 s and finished cleanly.
- **Daily limit: one submission per day** (the CLI reported "0 submissions remaining today" right after this one).
  Deadline 2026-11-02 23:59, so at most the slots of Oct 31, Nov 1 and Nov 2, and a nine-hour run must start early enough.
- Expected score is near 0; the goal of this submission is to exercise the real gateway path.

## How the score works, and the random floor (measured locally, 2026-10-31)

Read off the official scorecard of offline runs on the 24 public games (`scripts/local_play.py --fake`, a free scripted
client that plays random 1 to 4 move plans):

- **Level score = 100 x (baseline_actions / actions_used)^2**, capped at 100. For `sp80` level 1 (baseline 39): 43 actions
  scored 82.26, 46 scored 71.88, 82 scored 22.62, 134 scored 8.47. Actions used on the level include every wasted move.
- **Game score = mean of level scores weighted by level number** (level i has weight i). One cleared level 1 in a 6-level
  game is worth 100 x 1/21 = 4.8 at most. Overall score = mean over games.
- **Random-plan floor** (24 games, 150 to 200 steps each, 6 runs): 0.163, 0.143, 0.020, 0.0, 0.017, 0.056, mean about 0.067.
  Nearly all of it is luck on `sp80`, `vc33` and `ft09` level 1. Our submitted baseline (submission 57788190) scored 0.00.

Consequence: wall time is not the scarce resource, actions are. A model that thinks for a minute and then clears level 1 in
about the human baseline scores more than a fast one that flails. Next experiment: thinking ON with a large token budget
(kernel `leejianrong/arc3-duck-qwen-dev-think`), against thinking OFF (kernel `leejianrong/arc3-duck-qwen`).
