from __future__ import annotations

import logging

from datasets import Dataset, DatasetDict, load_dataset

from .config import DataConfig

logger = logging.getLogger(__name__)


def load_train_eval(cfg: DataConfig, seed: int) -> tuple[Dataset, Dataset]:
    """Return (train, eval) datasets.

    Prefers an existing ``validation`` (then ``test``) split. If neither
    exists, carves an eval split out of ``train`` deterministically.
    """
    dataset = load_dataset(cfg.dataset_id)
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    if "validation" in dataset:
        logger.info("Using existing 'validation' split for evaluation.")
        return dataset["train"], dataset["validation"]

    if "test" in dataset:
        logger.info("Using existing 'test' split for evaluation.")
        return dataset["train"], dataset["test"]

    logger.info(
        "No validation/test split found; carving %.0f%% out of train (seed=%d).",
        cfg.eval_split_size * 100,
        seed,
    )
    splits = dataset["train"].shuffle(seed=seed).train_test_split(
        test_size=cfg.eval_split_size, seed=seed
    )
    return splits["train"], splits["test"]
