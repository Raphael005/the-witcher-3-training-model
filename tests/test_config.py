"""Smoke tests for config loading — these run without GPU or network."""

from __future__ import annotations

from pathlib import Path

from witcher_sft.config import Config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "sft_qwen35_08b.yaml"


def test_loads_default_config():
    cfg = Config.from_yaml(CONFIG_PATH)
    assert cfg.model.model_id == "Qwen/Qwen3.5-0.8B"
    assert cfg.data.dataset_id
    assert cfg.seed == 42
    assert 0.0 < cfg.data.eval_split_size < 1.0


def test_output_dir_combines_root_and_run_name():
    cfg = Config.from_yaml(CONFIG_PATH)
    out = cfg.output.output_dir
    assert out.name == cfg.output.run_name


def test_output_dir_respects_env_override(monkeypatch):
    monkeypatch.setenv("SFT_OUTPUT_ROOT", "custom_runs")
    cfg = Config.from_yaml(CONFIG_PATH)
    assert cfg.output.output_dir.parts[0] == "custom_runs"
