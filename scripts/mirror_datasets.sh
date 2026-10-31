#!/usr/bin/env bash
# Private mirrors of the two third-party Kaggle datasets the notebook mounts (ARC-111), so the final run does
# not depend on anyone else keeping them alive (the original wheelhouse was deleted and re-created once).
# Downloads into fresh directories, verifies every byte against the pinned hashes with the notebook's own
# preflight, and only then uploads private copies under the SAME dataset names (kaggle/run/env.py finds
# datasets by name, so a notebook that attaches the mirrors works unchanged).
#   scripts/mirror_datasets.sh <workdir>
set -euo pipefail
cd "$(dirname "$0")/.."
WORK="${1:?usage: mirror_datasets.sh <empty workdir>}"
MODEL_SRC="driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
WHEEL_SRC="mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"
MODEL_DST="leejianrong/vrfai-qwen3-6-27b-fp8-hf-snapshot"
WHEEL_DST="leejianrong/arc3-vllm-h100-wheelhouse-v3"
mkdir -p "$WORK/model" "$WORK/wheelhouse"
[ -z "$(find "$WORK" -type f | head -1)" ] || { echo "workdir must hold no files" >&2; exit 1; }
kaggle datasets download -d "$MODEL_SRC" -p "$WORK/model" --unzip -q &
kaggle datasets download -d "$WHEEL_SRC" -p "$WORK/wheelhouse" --unzip -q &
wait
find "$WORK" -maxdepth 2 -name '*.zip' -delete
echo "downloaded: $(du -sh "$WORK/model" "$WORK/wheelhouse" | tr '\n' ' ')"
uv run solve-arc-agi-3 preflight --manifest configs/baselines/duck-control-v1.json \
  --model-dir "$WORK/model" --wheelhouse-dir "$WORK/wheelhouse" --full
echo "preflight passed: both copies match the pinned hashes"
for pair in "model|$MODEL_DST|vrfai Qwen3.6-27B-FP8 snapshot (private mirror)" "wheelhouse|$WHEEL_DST|arc3 vLLM wheelhouse v3 (private mirror)"; do
  IFS='|' read -r dir slug title <<<"$pair"
  printf '{"title": "%s", "id": "%s", "licenses": [{"name": "Apache 2.0"}], "isPrivate": true}\n' "$title" "$slug" >"$WORK/$dir/dataset-metadata.json"
  if kaggle datasets status "$slug" >/dev/null 2>&1; then
    kaggle datasets version -p "$WORK/$dir" -m "mirror refresh"
  else
    kaggle datasets create -p "$WORK/$dir"
  fi
done
echo "mirrors uploaded: $MODEL_DST $WHEEL_DST"
