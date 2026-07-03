#!/usr/bin/env python
"""Thin wrapper so you can run training without installing the package.

Usage:
    python scripts/train.py --config configs/sft_qwen35_08b.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable when running from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from witcher_sft.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
