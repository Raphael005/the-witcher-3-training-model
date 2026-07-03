# The Witcher 3 Training Model

🎮 **Fine-tuned LLM for Witcher 3 conversations in Brazilian Portuguese**

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/tendrivalentin/witcher3-qwen35-08b-sft-ptbr)

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("tendrivalentin/witcher3-qwen35-08b-sft-ptbr")
tokenizer = AutoTokenizer.from_pretrained("tendrivalentin/witcher3-qwen35-08b-sft-ptbr")

messages = [{"role": "user", "content": "Quem é Ciri?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = model.generate(tokenizer(text, return_tensors="pt").input_ids, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Overview

Supervised fine-tuning (SFT) pipeline for **Qwen3.5-0.8B** on the
[`FelippeTN/witcher3-dataset-ptbr`](https://huggingface.co/datasets/FelippeTN/witcher3-dataset-ptbr)
dataset — a Witcher 3 instruction dataset in Brazilian Portuguese with ~547 chat examples.

The training code is configuration-driven: hyperparameters live in YAML files
under [`configs/`](configs/), and the Python package under
[`src/witcher_sft/`](src/witcher_sft/) reads them.

## Project layout

```
.
├── configs/                  # YAML training configs (the knobs you tune)
│   └── sft_qwen35_08b.yaml
├── src/witcher_sft/          # Importable package
│   ├── config.py             # Typed config loaded from YAML
│   ├── data.py               # Dataset loading + train/eval split
│   ├── model.py              # Model & tokenizer construction
│   ├── trainer.py            # Wires everything into TRL's SFTTrainer
│   └── cli.py                # `witcher-sft` entrypoint
├── scripts/train.py          # Run without installing the package
├── tests/                    # Fast, offline smoke tests
├── pyproject.toml            # Packaging + dependencies + tooling
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # installs the package + dev tools
cp .env.example .env               # then fill in HF_TOKEN if needed
```

> `bitsandbytes` (used by the `adamw_bnb_8bit` optimizer) requires a CUDA GPU.
> On CPU/macOS, switch `optim` in the config to `adamw_torch`.

## Training

```bash
# Installed entrypoint:
witcher-sft --config configs/sft_qwen35_08b.yaml

# Or, without installing:
python scripts/train.py --config configs/sft_qwen35_08b.yaml
```

Checkpoints and the final model are written to
`runs/<run_name>/` (override the location with the `SFT_OUTPUT_ROOT`
environment variable).

### Hardware

The default config is tuned for a single **RTX 4060 (8 GB VRAM)** doing full
fine-tuning of the 0.8B model:

- `per_device_train_batch_size: 2` × `gradient_accumulation_steps: 8`
  → effective batch of **16** without exhausting VRAM.
- `gradient_checkpointing` and the 8-bit optimizer (`adamw_bnb_8bit`) keep the
  static footprint around ~5 GB, leaving room for activations.
- `auto_find_batch_size` halves the per-device batch automatically if you still
  hit an out-of-memory error.

If VRAM is tight, lower `training.max_length` (e.g. 768 or 512). On a 16 GB card
(RTX 4060 Ti 16 GB) you can raise `per_device_train_batch_size` to 8 and set
`gradient_accumulation_steps` to 2.

## Configuration

Everything reproducible is captured in the YAML config. Key fields:

| Field | Meaning |
| --- | --- |
| `model.model_id` | Base model on the Hugging Face Hub |
| `data.dataset_id` | SFT dataset id |
| `data.eval_split_size` | Held-out fraction when the dataset has no eval split |
| `output.run_name` | Subfolder under the output root |
| `training.*` | Epochs, batch size, LR schedule, eval/save cadence, etc. |

To create a new experiment, copy the YAML, change `output.run_name`, tweak
hyperparameters, and point `--config` at it.

## Tests

```bash
pytest
```

The tests only exercise config loading, so they run quickly and need neither a
GPU nor network access.

## Training Results

Trained on Apple Silicon (MPS backend) for ~35 minutes:

| Metric | Initial | Final |
| --- | --- | --- |
| Train Loss | 2.957 | 0.660 |
| Train Accuracy | 46.78% | 85.07% |
| Eval Loss (best) | - | 2.391 |
| Eval Accuracy | - | 53.93% |

See [`TRAINING_REPORT.md`](TRAINING_REPORT.md) for detailed metrics and analysis.

## Inference Examples

The fine-tuned model generates contextually relevant Witcher 3 responses in Portuguese:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("runs/sft-witcher3-ptbr-qwen35-08b-full")
tokenizer = AutoTokenizer.from_pretrained("runs/sft-witcher3-ptbr-qwen35-08b-full")

messages = [{"role": "user", "content": "Quem é Ciri?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Sample outputs:**

| Prompt | Response |
| --- | --- |
| "Quem é Ciri?" | "Ciri é uma jovem feiticeira com espírito leshen, poderosa e tem um vínculo profundo com a floresta." |
| "O que são os sinais de um bruxo?" | "Sinais incluem: mutação genética, perfeitidão mágica, comportamento socialmente inusitado..." |
| "Conte-me sobre a Caçada Selvagem." | "A Caçada Selvagem é um grupo de elfos que se estabelecem em Velen, lutam contra o mal..." |

## macOS / MPS Compatibility

This project includes fixes for running on Apple Silicon:

- `tf32` is automatically disabled on non-CUDA devices
- Default optimizer changed to `adamw_torch` (no bitsandbytes dependency)
- Reduced batch size and sequence length for memory constraints

To run on MPS with limited memory:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python scripts/train.py
```
