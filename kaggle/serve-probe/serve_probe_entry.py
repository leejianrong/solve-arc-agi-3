"""Kernel entry: run the serve probe from the mounted source bundle."""
import runpy
import sys
from glob import glob

bundle = sorted(glob("/kaggle/input/*/*/solve-arc-agi-3-src") + glob("/kaggle/input/solve-arc-agi-3-src"))[0]
sys.path.insert(0, f"{bundle}/run")
runpy.run_path(f"{bundle}/run/serve_probe.py", run_name="__main__")
