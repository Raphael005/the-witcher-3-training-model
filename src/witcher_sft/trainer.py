from __future__ import annotations

import logging

import torch
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from .config import Config
from .data import load_train_eval
from .model import load_model, load_tokenizer, log_parameter_counts

logger = logging.getLogger(__name__)


def build_sft_config(cfg: Config) -> SFTConfig:
    """Translate our typed Config into a TRL SFTConfig."""
    t = cfg.training
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()

    return SFTConfig(
        output_dir=str(cfg.output.output_dir),
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        auto_find_batch_size=t.auto_find_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        warmup_steps=t.warmup_steps,
        weight_decay=t.weight_decay,
        lr_scheduler_type=t.lr_scheduler_type,
        optim=t.optim,
        max_length=t.max_length,
        bf16=bf16,
        fp16=cuda and not bf16,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        assistant_only_loss=t.assistant_only_loss,
        loss_type=t.loss_type,
        do_train=True,
        do_eval=True,
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        logging_steps=t.logging_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        report_to=t.report_to,
        remove_unused_columns=False,
        seed=cfg.seed,
        data_seed=cfg.seed,
        tf32=cuda,
        use_cache=False,
    )


def run_training(cfg: Config) -> dict:
    cfg.output.output_dir.parent.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    train_dataset, eval_dataset = load_train_eval(cfg.data, cfg.seed)
    logger.info("Train/eval examples: %d / %d", len(train_dataset), len(eval_dataset))

    tokenizer = load_tokenizer(cfg.model)
    model = load_model(cfg.model)
    log_parameter_counts(model)

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(cfg),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()
    logger.info("Training metrics: %s", result.metrics)

    output_dir = str(cfg.output.output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved to: %s", output_dir)

    return result.metrics
