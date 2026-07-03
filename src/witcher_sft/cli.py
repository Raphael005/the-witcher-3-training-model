from __future__ import annotations

import argparse
import logging

from .config import Config
from .trainer import run_training


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="witcher-sft",
        description="Supervised fine-tuning of Qwen3.5 on the Valdoria dataset.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/sft_qwen35_08b.yaml",
        help="Path to a YAML training config (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _setup_logging()
    args = parse_args(argv)
    cfg = Config.from_yaml(args.config)
    logging.getLogger(__name__).info("Loaded config from %s", args.config)
    run_training(cfg)


if __name__ == "__main__":
    main()
