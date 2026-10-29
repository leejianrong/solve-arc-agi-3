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
| vLLM wheelhouse | `driessmit1/arc3-vllm-h100-wheelhouse-v3` | Pinned in duck-control-v1; **check it supports the competition GPU (RTX PRO 6000 Blackwell)** |
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
