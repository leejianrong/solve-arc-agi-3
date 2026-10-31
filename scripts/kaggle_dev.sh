#!/usr/bin/env bash
# One-command variant run on Kaggle (free GPU): publish HEAD, push a dev notebook that sets the given
# ARC3_* variables, wait for it, and print a summary computed from its recordings and trace.
#   scripts/kaggle_dev.sh <name> [KEY=VALUE ...]      e.g. scripts/kaggle_dev.sh ledger ARC3_LEDGER=1
# Needs the kaggle CLI. At most two GPU kernels run at once on Kaggle. Results land in build/dev-runs/<name>/.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?usage: kaggle_dev.sh <name> [KEY=VALUE ...]}"; shift
SLUG="leejianrong/arc3-dev-${NAME}"
DIR="build/kaggle-dev-${NAME}"
OUT="build/dev-runs/${NAME}"
rm -rf "$DIR" && mkdir -p "$DIR" "$OUT"
python3 - "$DIR" "$SLUG" "$@" <<'PY'
import json, sys
d, slug, *pairs = sys.argv[1:]
env = dict(p.split("=", 1) for p in pairs)
env.setdefault("ARC3_BUNDLE_SLUG", "solve-arc-agi-3-dev")
env.setdefault("ARC3_DRY_GAMES", "12")
env.setdefault("ARC3_PLAY_SECONDS", str(40 * 60))
src = ["import os, runpy, sys\n", "from glob import glob\n", f"os.environ.update({env!r})\n",
       "bundle = min(glob('/kaggle/input/datasets/*/solve-arc-agi-3-dev') + glob('/kaggle/input/solve-arc-agi-3-dev'))\n",
       "sys.path.insert(0, bundle + '/run')\n", "runpy.run_path(bundle + '/run/main.py', run_name='__main__')\n"]
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(f"{d}/nb.ipynb", "w"), indent=1)
meta = json.load(open("kaggle/submission/kernel-metadata.json"))
meta["dataset_sources"] = [x.replace("solve-arc-agi-3-src", "solve-arc-agi-3-dev") for x in meta["dataset_sources"]]
meta.update(id=slug, title=slug.split("/")[1].replace("-", " "), code_file="nb.ipynb")
json.dump(meta, open(f"{d}/kernel-metadata.json", "w"), indent=2)
PY
BUNDLE_SLUG=leejianrong/solve-arc-agi-3-dev scripts/build_kaggle_bundle.sh "dev ${NAME}: $(git rev-parse --short HEAD)" >&2
kaggle kernels push -p "$DIR" >&2
until s=$(kaggle kernels status "$SLUG" 2>&1 | tail -1); case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) true;; *) false;; esac; do sleep 45; done
echo "$s" >&2
kaggle kernels output "$SLUG" -p "$OUT" >/dev/null 2>&1 || true
python3 scripts/summarize_run.py "$OUT"
