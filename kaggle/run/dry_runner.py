"""Dry run: play a few public games offline through the official Swarm and print the scorecard."""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(levelname)s | %(message)s")

from agents import Swarm  # noqa: E402

env_dir = Path(os.environ["ENVIRONMENTS_DIR"])
games = sorted(json.loads(p.read_text())["game_id"] for p in env_dir.glob("*/*/metadata.json"))
want = int(os.environ.get("ARC3_DRY_GAMES", "6"))
games = games[:want]
print("dry-run games:", games, flush=True)
card = Swarm("myagent", "http://unused", games).main()
if card is not None:
    print("SCORECARD", json.dumps(card.model_dump(), default=str)[:6000], flush=True)
