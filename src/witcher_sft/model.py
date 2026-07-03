from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import ModelConfig

logger = logging.getLogger(__name__)


def load_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(cfg: ModelConfig) -> PreTrainedModel:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
    )
    # use_cache must be off when gradient checkpointing is on.
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def log_parameter_counts(model: PreTrainedModel) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0.0
    logger.info("Trainable parameters: %s (%.2f%%)", f"{trainable:,}", pct)
