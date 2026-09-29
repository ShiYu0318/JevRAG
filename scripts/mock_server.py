"""Fake /v1/systemone server for plumbing tests.

    python scripts/mock_server.py --port 8765
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevrag.mock import serve  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"mock /v1/systemone on :{a.port}")
    serve(a.port)
