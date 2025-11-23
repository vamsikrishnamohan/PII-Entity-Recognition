# PII NER Assignment Skeleton

This repo is a skeleton for a token-level NER model that tags PII in STT-style transcripts.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python3 src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \
  --epochs 5 \
  --batch_size 8 \
  --lr 5e-5 \
  --patience 3 \
  --grad_clip 1.0
```

**New Training Features:**
- **Dev set validation** after each epoch
-  **Early stopping** based on dev loss (configurable patience)
-  **Best model saving** - automatically saves the model with lowest dev loss
-  **Gradient clipping** for training stability
-  **Fast tokenizer** for better performance

The training script now saves two models:
- `out/best_model/` - Best performing model on dev set (use this for inference)
- `out/final_model/` - Model from the last epoch

## Predict

```bash
python3 src/predict.py \
  --model_dir out/best_model \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

**Note:** Use `out/best_model` (best performing on dev set) for predictions.

## Evaluate

```bash
python3 src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

## Measure latency

```bash
python src/measure_latency.py \
  --model_dir out/best_model \
  --input data/dev.jsonl \
  --runs 50
```

Your task in the assignment is to modify the model and training code to improve entity and PII detection quality while keeping **p95 latency below ~20 ms** per utterance (batch size 1, on a reasonably modern CPU).
