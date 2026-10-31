# duck_local: Franzen's Duck harness, offline, text only

Plays the public games (`../play-arc-agi-3/environment_files`) with his patched Tufa Duck harness against any
OpenAI-compatible endpoint (a semula project, or `stub_server.py`), so prompts and observation text can be
studied. Model-behaviour checks stay on semula + Haiku (`docs/SEMULA-HAIKU-LOOP.md`); no hosted API.

```sh
# semula: one fresh project per run, then Haiku players answer the parked calls
docker exec semula-arc3-semula-1 semula project create duck-r1
BASE_URL=http://127.0.0.1:8080/v1/projects/duck-r1 GAMES=ls20 MAX_ACTIONS=20 scripts/duck_local/run.sh

# free smoke test with the scripted stub (answers every call with action("UP"), "RIGHT", ...)
python3 scripts/duck_local/stub_server.py 18099 /tmp/stub.jsonl &
BASE_URL=http://127.0.0.1:18099/v1 MAX_ACTIONS=6 MAX_RUNTIME=120 scripts/duck_local/run.sh
```

First run builds `build/duck_local/` (gitignored): pulls his notebook (`kaggle kernels pull`) and the source
bundle (`kaggle datasets download dfranzen/taaf-kaggle-source-bundle-copy`), extracts the `%%writefile` patch
from the notebook JSON, `git apply --include='ARC3-Inference/*'`, and makes a Python 3.12 venv from the bundle's
`uv.lock`. Later runs reuse all of it. Artifacts and transcripts go to `$OUT` (`transcripts/*.txt`,
`requests.jsonl`, `prompts/`, `artifacts/*_events.jsonl`, `diagnostics.html`, `movies/`).

| Env var | Default | Meaning |
| --- | --- | --- |
| `BASE_URL` | required | OpenAI base URL ending before `/chat/completions` |
| `GAMES` | `ls20` | comma list of game id prefixes |
| `MAX_ACTIONS` | 20 | per-game action cap (the warmup RESET counts) |
| `MAX_RUNTIME` | 300 | seconds per game |
| `CONCURRENCY` | 1 | games in parallel (one request each) |
| `MODEL` | `flashnext` | model id sent in requests |
| `OUT`, `WORK` | `build/duck_local/out/<time>`, `build/duck_local` | output and cache dirs |
| `ENV_DIR`, `NOTEBOOK`, `BUNDLE_SRC` | sibling env files, pulled copies | override to reuse local copies (`BUNDLE_SRC` is an unpacked bundle) |

The harness switches (his notebook cell 4) live in `HARNESS_ENV` in `duck_local.py`. Differences: `MULTIMODAL_CONTEXT=off`,
`ARC3_DIFF_IMAGE=0`, `ARC3_GAMEOVER_DIFF_IMAGE=0` (no images); no priority scheduler; no vLLM/sglang.

Known gaps:
- Text only: his vision prompts and the diff images are not exercised.
- `arc-agi-3-local` (`re_arc`, private repo) is not installed; the offline path does not import it.
- The venv skips `arc-agi` from his wheelhouse and uses the lockfile's `arc-agi==0.9.9`.
- The warmup RESET fires before the first request (`ARC3_WARMUP_ACTION_GAMES=1`), so N actions means N-1 model calls.
- Each request carries the whole history, so `requests.jsonl` and a stub log grow fast; keep `MAX_ACTIONS` small.
- Running his pickles (`benchmark_initial.pkl`) executes code: only use the bundle dataset you trust.
