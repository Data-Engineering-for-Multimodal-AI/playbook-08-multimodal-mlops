#!/usr/bin/env python
"""Wrapper kept so the printed command `python pipeline/run_pipeline.py` keeps working.

The first edition kept a full copy of the code in this folder. There is now one copy, at the
repository root; this script runs the root `run_pipeline.py` from the root folder, wherever
it is started from, and passes on any arguments.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, os.path.join(ROOT, "run_pipeline.py"), *sys.argv[1:]], cwd=ROOT))
