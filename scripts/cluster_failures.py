#!/usr/bin/env python3
"""Rank why games were not cleared, from a step trace folder (docs/TRACE.md).

scripts/cluster_failures.py traces/<run_id> [more run folders]
"""

import sys
from pathlib import Path

from solve_arc_agi_3.failure_clusters import load_run, report

if __name__ == "__main__":
    folders = [Path(a) for a in sys.argv[1:]]
    if not folders:
        raise SystemExit(__doc__)
    print(report([v for f in folders for v in load_run(f)]))
