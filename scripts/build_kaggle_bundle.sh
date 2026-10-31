#!/usr/bin/env bash
# Build and publish the private Kaggle dataset that carries our own code
# (package, control manifest, run scripts) into the offline notebook.
# Usage: scripts/build_kaggle_bundle.sh [version-notes]
# Needs the kaggle CLI logged in. Contains no secrets and no third-party code.
set -euo pipefail
cd "$(dirname "$0")/.."
SLUG="leejianrong/solve-arc-agi-3-src"
OUT="build/kaggle-src"
# Ship the committed HEAD only, never the working tree: another agent (or a half-finished edit) can
# be mid-change in the tree, and GIT_COMMIT below must describe exactly what is in the bundle.
if ! git diff --quiet HEAD -- src configs/baselines kaggle/run; then
  echo "note: uncommitted changes under src/, configs/baselines or kaggle/run are NOT in this bundle" >&2
fi
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
rm -rf "$OUT" && mkdir -p "$OUT/configs"
git archive HEAD src/solve_arc_agi_3 configs/baselines kaggle/run LICENSE | tar -x -C "$STAGE"
cp -r "$STAGE/src/solve_arc_agi_3" "$OUT/solve_arc_agi_3"
cp -r "$STAGE/configs/baselines" "$OUT/configs/baselines"
cp -r "$STAGE/kaggle/run" "$OUT/run"
cp "$STAGE/LICENSE" "$OUT/LICENSE"
cat >"$OUT/dataset-metadata.json" <<JSON
{"title": "solve-arc-agi-3 source", "id": "$SLUG", "licenses": [{"name": "Apache 2.0"}]}
JSON
git rev-parse HEAD >"$OUT/GIT_COMMIT" 2>/dev/null || echo unknown >"$OUT/GIT_COMMIT"
if kaggle datasets status "$SLUG" >/dev/null 2>&1; then
  kaggle datasets version -p "$OUT" -m "${1:-update}" --dir-mode zip
else
  kaggle datasets create -p "$OUT" --dir-mode zip
fi

# A kernel started before the new version is visible would silently run stale code: wait until the published
# GIT_COMMIT matches what was just built.
want="$(cat "$OUT/GIT_COMMIT")"
tmp="$(mktemp -d)"
for _ in $(seq 1 60); do
  rm -f "$tmp"/GIT_COMMIT*
  kaggle datasets download "$SLUG" -f GIT_COMMIT -p "$tmp" --force -q >/dev/null 2>&1 || true
  [ -f "$tmp/GIT_COMMIT.zip" ] && unzip -qo "$tmp/GIT_COMMIT.zip" -d "$tmp" 2>/dev/null
  if [ -f "$tmp/GIT_COMMIT" ] && [ "$(cat "$tmp/GIT_COMMIT")" = "$want" ]; then echo "bundle published: $want"; exit 0; fi
  sleep 5
done
echo "bundle version with commit $want not visible after 5 minutes" >&2
exit 1
