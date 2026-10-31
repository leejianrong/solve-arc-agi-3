#!/usr/bin/env bash
# Franzen's Duck harness, offline, text only, against an OpenAI-compatible endpoint (semula or the stub).
# See README.md. Required: BASE_URL. Everything else has a default.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
WORK=${WORK:-$REPO/build/duck_local}           # venv, notebook, bundle copy (reused between runs)
OUT=${OUT:-$WORK/out/$(date +%Y%m%d-%H%M%S)}   # run artifacts and transcripts
BASE_URL=${BASE_URL:?set BASE_URL, e.g. http://127.0.0.1:8080/v1/projects/<slug>}
GAMES=${GAMES:-ls20}                           # comma list of game id prefixes
MAX_ACTIONS=${MAX_ACTIONS:-20}
MAX_RUNTIME=${MAX_RUNTIME:-300}                # seconds per game
CONCURRENCY=${CONCURRENCY:-1}
MODEL=${MODEL:-flashnext}
ENV_DIR=${ENV_DIR:-$REPO/../play-arc-agi-3/environment_files}
NOTEBOOK=${NOTEBOOK:-$WORK/nb/arc-agi-3-milestone-2-solution.ipynb}
BUNDLE_SRC=${BUNDLE_SRC:-$WORK/bundle_orig}    # unpacked dfranzen/taaf-kaggle-source-bundle-copy
BUNDLE=$WORK/bundle
VENV=$WORK/venv
mkdir -p "$WORK"

[ -f "$NOTEBOOK" ] || kaggle kernels pull dfranzen/arc-agi-3-milestone-2-solution -p "$(dirname "$NOTEBOOK")" -m
[ -d "$BUNDLE_SRC/src" ] || kaggle datasets download dfranzen/taaf-kaggle-source-bundle-copy --unzip -p "$BUNDLE_SRC"

if [ ! -f "$BUNDLE/.patched" ]; then
  rm -rf "$BUNDLE" && cp -a "$BUNDLE_SRC" "$BUNDLE"
  python3 -I "$HERE/duck_local.py" extract-patch "$NOTEBOOK" "$WORK/harness-changes.patch"
  # His notebook applies exactly this; git apply fails loudly (set -e) if it does not apply.
  (cd "$BUNDLE/src" && git apply --include='ARC3-Inference/*' -v "$WORK/harness-changes.patch")
  grep -q reset_exposed "$BUNDLE/src/ARC3-Inference/inference/agent/action_names.py"  # patch really landed
  touch "$BUNDLE/.patched"
fi

if [ ! -x "$VENV/bin/python" ]; then
  # re-arc-3 (arc-agi-3-local) is a private repo and the offline path does not import it, so it is left out;
  # the two bundled repos go on sys.path through a .pth, as in his notebook.
  (cd "$BUNDLE/src/ARC3-Inference" && uv export --frozen --no-hashes --no-dev \
     --no-emit-package arc-agi-3-local --no-emit-package tufa-arc-agi-framework --no-emit-package arc3-inference \
     > "$WORK/requirements.txt")
  uv venv --python 3.12 "$VENV"
  uv pip install -p "$VENV/bin/python" -r "$WORK/requirements.txt"
  SP=$("$VENV/bin/python" -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
  printf '%s\n' "$BUNDLE/src/tufa-arc-agi-framework/src" "$BUNDLE/src/tufa-arc-agi-framework" \
    "$BUNDLE/src/ARC3-Inference" > "$SP/taaf_sources.pth"
fi

mkdir -p "$OUT"
echo "out: $OUT"
exec "$VENV/bin/python" "$HERE/duck_local.py" play --bundle "$BUNDLE" --out "$OUT" --env-dir "$ENV_DIR" \
  --games "$GAMES" --base-url "$BASE_URL" --model "$MODEL" --max-actions "$MAX_ACTIONS" \
  --max-runtime "$MAX_RUNTIME" --concurrency "$CONCURRENCY"
