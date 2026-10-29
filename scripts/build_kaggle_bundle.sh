#!/usr/bin/env bash
# Build and publish the private Kaggle dataset that carries our own code
# (package, control manifest, run scripts) into the offline notebook.
# Usage: scripts/build_kaggle_bundle.sh [version-notes]
# Needs the kaggle CLI logged in. Contains no secrets and no third-party code.
set -euo pipefail
cd "$(dirname "$0")/.."
SLUG="leejianrong/solve-arc-agi-3-src"
OUT="build/kaggle-src"
rm -rf "$OUT" && mkdir -p "$OUT/configs"
cp -r src/solve_arc_agi_3 "$OUT/solve_arc_agi_3"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
cp -r configs/baselines "$OUT/configs/baselines"
cp -r kaggle/run "$OUT/run"
cp LICENSE "$OUT/LICENSE"
cat >"$OUT/dataset-metadata.json" <<JSON
{"title": "solve-arc-agi-3 source", "id": "$SLUG", "licenses": [{"name": "Apache 2.0"}]}
JSON
git rev-parse HEAD >"$OUT/GIT_COMMIT" 2>/dev/null || echo unknown >"$OUT/GIT_COMMIT"
if kaggle datasets status "$SLUG" >/dev/null 2>&1; then
  kaggle datasets version -p "$OUT" -m "${1:-update}" --dir-mode zip
else
  kaggle datasets create -p "$OUT" --dir-mode zip
fi
