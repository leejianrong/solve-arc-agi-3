#!/usr/bin/env python3
"""Run Franzen's Duck harness offline on the public games against any OpenAI-compatible endpoint.

Two subcommands: `extract-patch NOTEBOOK OUT` (pure, no harness needed) and `play` (runs inside the venv that
run.sh builds, because it imports taaf and unpickles the trusted bundle).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Notebook cell 4 (setup_env), minus serving, vision and scheduler keys, which are set below.
HARNESS_ENV = {
    "ARC3_REASONING_HISTORY_KEY": "reasoning_content",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_YIELD_SECONDS": "0",
    "LOCAL_ANALYZER_YIELD_TOKENS": "2048",
    "LOCAL_ANALYZER_TEMPERATURE": "0.7",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": str((116 + 12) * 1024),
    "LOCAL_ANALYZER_MAX_OUTPUT": str(12 * 1024),
    "ARC3_CONTEXT_DRAIN_TOKENS": str(58 * 1024),
    "ARC3_HISTORY_ASSISTANT_TURNS": "150",
    "ARC3_HISTORY_TURN_DRAIN": "30",
    "ARC3_HISTORY_DRAIN_COALESCE": "1",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "3072",
    "ARC3_MIDDLE_TRUNCATION": "1",
    "ARC3_ACTION_ECHO": "1",
    "ARC3_EXPLAIN_GAMEPLAY_CHANGED": "1",
    "ARC3_NEW_CHANGED_PROMPTS": "1",
    "ARC3_LEVEL_TRANSFER_GUIDANCE": "1",
    "ARC3_ACTION_INFO": "1",
    "ARC3_DEDUPE_MULTICALL_LINE": "1",
    "ARC3_FRAME_DIFF_HINT": "1",
    "ARC3_ANIMATION": "1",
    "ARC3_GUARDS_FROM_LEVEL": "2",
    "ARC3_BATCH_NOOP_BLOCK": "1",
    "ARC3_STALE_STATE_BLOCK": "1",
    "ARC3_MEMORY_SECTIONS": "off",
    "ARC3_PERSISTENT_FUNCTIONS": "1",
    "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
    "ARC3_PERSISTENT_FUNCTIONS_IMPORTS": "1",
    "EXPOSE_UNDO": "on",
    "ARC3_WARMUP_ACTION_GAMES": "1",
    # Text only: anything but "current_grid" turns the grid image off; the diff images are separate switches.
    "MULTIMODAL_CONTEXT": "off",
    "ARC3_DIFF_IMAGE": "0",
    "ARC3_GAMEOVER_DIFF_IMAGE": "0",
    "ONLY_RESET_LEVELS": "true",
    "MPLBACKEND": "Agg",
    "TAAF_RUN_AS_SUBMISSION": "0",
    "TAAF_MINIMAL_DIAGNOSTICS": "0",
}


def extract_patch(notebook: Path) -> str:
    """The `%%writefile ...harness-changes.patch` cell of his notebook, without the magic line."""
    for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
        text = "".join(cell["source"])
        magic, _, body = text.partition("\n")
        if (
            cell["cell_type"] == "code"
            and magic.startswith("%%writefile")
            and magic.endswith(".patch")
        ):
            return body if body.endswith("\n") else body + "\n"
    raise ValueError(f"no %%writefile .patch cell in {notebook}")


def play(args: argparse.Namespace) -> None:
    import arc_agi
    import taaf.game_api

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")
    os.environ.update(HARNESS_ENV)
    os.environ.update(
        {
            "OPENAI_BASE_URL": base_url,
            "LOCAL_ANALYZER_BASE_URL": base_url,
            "OPENAI_PROVIDER": "vllm",
            "LOCAL_ANALYZER_PROVIDER": "vllm",
            "LOCAL_ANALYZER_MODEL_ID": args.model,
            "INFERENCE_ANALYZER_MODEL": args.model,
            "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
            "OPENAI_API_KEY": "none",
            "LOCAL_ANALYZER_API_KEY": "none",
            "RECORDINGS_DIR": str(out / "server_recording"),
        }
    )
    # The bundle is the user's trusted dataset; its pickles hold the solver config and the benchmark shell.
    bundle = Path(args.bundle)
    with open(bundle / "benchmark_initial.pkl", "rb") as fh:
        bm = pickle.load(fh)
    bm.job_dir = out
    bm.n_passes = 1
    bm.game_weights = None
    bm.solver.max_runtime_s_per_game = args.max_runtime
    bm.solver.analyzer_timeout = args.analyzer_timeout
    bm.solver.concurrency = args.concurrency
    bm.solver.max_actions_per_game = args.max_actions
    bm.solver.save_request_logs = True

    env_dir = str(Path(args.env_dir).resolve())
    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir
    )
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir
    )
    wanted = [g for g in args.games.split(",") if g]
    ids = [
        e.game_id
        for e in arcade.available_environments
        if any(e.game_id.startswith(g) for g in wanted)
    ]
    if not ids:
        sys.exit(f"no games match {wanted} under {env_dir}")
    bm.games = [taaf.game_api.GameAPI(env_name=i, arcade_spec=spec) for i in ids]
    print("games:", ids, "->", out, flush=True)
    soft_end = datetime.now(UTC) + timedelta(seconds=args.max_runtime + 60)
    asyncio.run(bm.run(soft_end_time=soft_end))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract-patch")
    ex.add_argument("notebook")
    ex.add_argument("out")
    p = sub.add_parser("play")
    p.add_argument(
        "--bundle",
        required=True,
        help="patched, unpacked bundle dir (has benchmark_initial.pkl)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--env-dir", required=True)
    p.add_argument("--games", default="ls20", help="comma list of game id prefixes")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="flashnext")
    p.add_argument("--max-actions", type=int, default=20)
    p.add_argument("--max-runtime", type=float, default=300)
    p.add_argument("--analyzer-timeout", type=float, default=900)
    p.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "extract-patch":
        Path(a.out).write_text(extract_patch(Path(a.notebook)), encoding="utf-8")
    else:
        play(a)


if __name__ == "__main__":
    main()
