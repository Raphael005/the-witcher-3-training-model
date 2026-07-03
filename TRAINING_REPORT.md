# Witcher 3 SFT Training Report

**Date:** 2026-07-03  
**Model:** Qwen3.5-0.8B (full fine-tune)  
**Dataset:** FelippeTN/witcher3-dataset-ptbr (Portuguese)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen3.5-0.8B |
| Trainable Parameters | 752,393,024 (100%) |
| Training Examples | 492 |
| Evaluation Examples | 55 |
| Epochs | 3 |
| Batch Size | 1 (effective: 16 with gradient accumulation) |
| Learning Rate | 1.5e-5 (cosine scheduler) |
| Max Sequence Length | 512 |
| Optimizer | AdamW (torch) |
| Precision | float32 (MPS) |

## Results

### Training Metrics

| Metric | Initial | Final |
|--------|---------|-------|
| Train Loss | 2.957 | 0.660 |
| Train Accuracy | 46.78% | 85.07% |

### Evaluation Metrics

| Checkpoint | Eval Loss | Eval Accuracy |
|------------|-----------|---------------|
| Step 25 | 2.463 | 51.59% |
| Step 50 (best) | 2.391 | 53.93% |
| Step 75 | 2.478 | 53.78% |
| Step 93 (final) | 2.554 | 54.44% |

### Training Curve Summary

- Loss decreased consistently from 2.96 → 0.66 over 3 epochs
- Best evaluation loss achieved at step 50 (epoch ~1.6)
- Slight overfitting observed after step 50 (eval loss increased while train loss continued decreasing)

## Model Artifacts

```
runs/sft-witcher3-ptbr-qwen35-08b-full/
├── model.safetensors      # Fine-tuned weights (~3GB)
├── config.json            # Model configuration
├── tokenizer.json         # Tokenizer
├── tokenizer_config.json  # Tokenizer settings
├── chat_template.jinja    # Chat format template
├── checkpoint-50/         # Best checkpoint
└── checkpoint-93/         # Final checkpoint
```

## Sample Outputs

**Prompt:** "Geralt, o que você acha dos monstros que assombram Velen?"  
**Response:** "Velen tem vários assombres, incluindo o Monstro da Caçada Selvagem e o Gato Mágico. Cada um de eles tem sua própria história e conexão com a cultura local, sendo importantes para entender a atmosfera da região."

**Prompt:** "Como você derrotaria um grifo?"  
**Response:** "O grifo é um monstro aquático poderoso. Para matá-lo, use um set de ataque rápido (Wing Sweep ou similar) que destruído os três grifos ao redor simultaneamente. Dimeritium Bomb pode eliminar alguns de eles."

**Prompt:** "Fale sobre Yennefer."  
**Response:** "Yennefer é uma poderosa feiticeira que domina a corte de Redânia. Sua influência é questionável, mas seu papel como agente de poder nos reinos do norte é amplamente reconhecido."

## Recommendations

1. **Use checkpoint-50** for best generalization (lowest eval loss)
2. **Increase dataset size** to reduce overfitting
3. **Consider LoRA/QLoRA** for more efficient fine-tuning with less overfitting
4. **Add early stopping** based on eval loss to prevent overfitting

## Environment

- Hardware: Apple Silicon (MPS backend)
- Runtime: ~35 minutes
- Total FLOPs: 5.73e+14
