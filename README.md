# embed_distance

Compute semantic distance between any two strings and visualise clusters of phrases — fully offline after a one-time model download.

**Model:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — 22 M parameters, 384-dim embeddings, ~80 MB on disk.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download model weights (internet required — run once)
python download_model.py

# From this point everything works fully offline
```

---

## Scripts

### `distance.py` — compare two strings

```bash
# Interactive REPL
python distance.py

# Direct comparison
python distance.py "machine learning" "deep learning"

# Batch: rank many phrases against a query
python distance.py --query "neural network" --file phrases.txt

# Verbose (shows embedding norms, debug info)
python distance.py -v "cat" "dog"
```

Output example:
```
Cosine similarity :  +0.821043  (1 = identical, 0 = unrelated, -1 = opposite)
Cosine distance   :   0.178957  (0 = identical, 2 = maximally distant)
Euclidean distance:   0.598612
```

---

### `visualize.py` — scatter plot of phrase clusters

```bash
# Use built-in sample phrases (AI/ML, Food, Sports, Finance)
python visualize.py

# Load your own phrases file and save PNGs
python visualize.py --file phrases.txt --save

# Skip t-SNE for speed (only PCA)
python visualize.py --no-tsne
```

Generated files: `embed_pca.png`, `embed_tsne.png`

---

## Phrases file format (`phrases.txt`)

```
# Lines starting with # are ignored
# Plain line → grouped under "Custom"
deep learning

# Labelled line → grouped by label in the plot
AI / ML::machine learning
Food::sushi roll
```

---

## Offline deployment

To move to a machine with no internet access:

1. Run `python download_model.py` on an online machine.
2. Copy the entire project directory (including `models/`) to the target machine.
3. Install Python deps from a local cache or wheel files:
   ```bash
   # On online machine — download wheels
   pip download -r requirements.txt -d ./wheels/
   # On offline machine — install from wheels
   pip install --no-index --find-links=./wheels/ -r requirements.txt
   ```
4. All three scripts run without any network access.

---

## Project layout

```
embed_distance/
├── download_model.py              # One-time online setup
├── distance.py                    # Pairwise / batch distance tool
├── visualize.py                   # Cluster scatter plots
├── validator.py                   # GPT response validator
├── train_refusal_classifier.py    # Train the refusal detector
├── phrases.txt                    # Sample phrases file
├── requirements.txt
├── data/
│   └── refusal_training.jsonl     # Labeled dataset (240 examples)
├── models/
│   ├── all-MiniLM-L6-v2/         # Sentence transformer weights
│   └── refusal_classifier.pt     # Trained classifier (after training)
└── README.md
```

---

## GPT Response Validator

`validator.py` compares two LLM responses against a known ground-truth answer
and produces a single **composite score** in `[-1, +1]`:

| Score | Label | Meaning |
|-------|-------|---------|
| `+1.0` | `dual_refusal` | Both responses refuse — they agree on refusing |
| `+1.0` | `agreement` | Both responses encode the same meaning |
| `(0.4, 1)` | `partial_divergence` | Responses diverge from each other or GT |
| `(0, 0.4)` | `divergence` | Responses are orthogonal; neither relates to GT |
| `-1.0` | `guardrail_conflict` | One response is substantive, the other refuses |

The score is computed as:
- **`-1.0`** when exactly one response is a refusal (XOR)
- **`+1.0`** when both responses are refusals (agreement on refusing)
- **cosine similarity** between the two responses when both are substantive

### Training the Refusal Classifier

The refusal detector is a small MLP head (384 → 64 → 1) trained on frozen
embeddings from `all-MiniLM-L6-v2`. The training dataset (`data/refusal_training.jsonl`)
contains 240 labeled examples in Spanish drawn from banking audit and Spanish
banking regulation Q&A.

```bash
# 1. Install dependencies (adds torch)
pip install -r requirements.txt

# 2. Inspect the dataset
head -5 data/refusal_training.jsonl

# 3. Train (~30 seconds on CPU)
python train_refusal_classifier.py

# 4. Verify the output
ls models/refusal_classifier.pt
cat training.log
```

Training options:
```
python train_refusal_classifier.py [--data data/refusal_training.jsonl]
                                    [--output models/refusal_classifier.pt]
                                    [--threshold 0.5]
                                    [--epochs 50] [--lr 1e-3] [-v]
```

Expected output at the end of training:
```
[INFO] Test accuracy : 97.5%
[INFO] Test F1 score : 0.975
[INFO] Confusion matrix:
       Pred 0  Pred 1
True 0   12       0
True 1    1      11
[INFO] Model saved → models/refusal_classifier.pt
```

### Validator usage

```bash
# Single comparison (3 positional args)
python validator.py "Respuesta A" "Respuesta B" "Texto de referencia"

# From files
python validator.py --file-a response_a.txt --file-b response_b.txt \
                    --gt ground_truth.txt

# Batch mode (JSONL with fields response_a, response_b, ground_truth)
python validator.py --batch comparisons.jsonl --output results.jsonl

# Adjust refusal threshold
python validator.py -t 0.6 "Respuesta A" "Respuesta B" "Referencia"
```

Example output (guardrail conflict):
```
Composite score   :  -1.000000  (+1=agree, 0=diverge, -1=guardrail conflict)
Case              :  guardrail_conflict
─────────────────────────────────────────────
A vs B similarity :  +0.123456
A vs GT similarity:  +0.821043
B vs GT similarity:  +0.031200
─────────────────────────────────────────────
Response A refusal:  No  (confidence: 0.97)
Response B refusal:  Sí  (confidence: 0.97)
```

### Dataset format

`data/refusal_training.jsonl` — one JSON object per line:

```jsonl
{"text": "No puedo proporcionar esa información.", "label": 0}
{"text": "La ratio LCR exige activos líquidos para cubrir salidas de 30 días.", "label": 1}
```

| Field | Type | Values |
|-------|------|--------|
| `text` | string | The response text |
| `label` | int | `0` = refusal / guardrail, `1` = substantive answer |

To extend the dataset, append new lines following this schema and re-run
`train_refusal_classifier.py`.

### Scoring interpretation guide

| Range | Label | Example |
|-------|-------|---------|
| `score = +1.0` and `dual_refusal` | Both refused | "No puedo ayudar" vs "No estoy autorizado" |
| `0.80 ≤ score ≤ 1.0` | `agreement` | Two paraphrases of the same regulatory fact |
| `0.40 ≤ score < 0.80` | `partial_divergence` | Related but diverging explanations |
| `score < 0.40` | `divergence` | Unrelated substantive answers |
| `score = -1.0` | `guardrail_conflict` | One factual answer vs one refusal |

### Architecture

```
                    ┌─────────────────────┐
 Response A ───────▶│                     │──▶ emb_a (384-d) ──▶ RefusalClassifier ──▶ is_refusal_a
                    │  all-MiniLM-L6-v2   │
 Response B ───────▶│  (frozen backbone)  │──▶ emb_b (384-d) ──▶ RefusalClassifier ──▶ is_refusal_b
                    │                     │
 Ground Truth ─────▶│                     │──▶ emb_gt (384-d)
                    └─────────────────────┘
                              │
               cosine_similarity(emb_a, emb_b) ──▶ sim_ab
               cosine_similarity(emb_a, emb_gt) ──▶ sim_a_gt
               cosine_similarity(emb_b, emb_gt) ──▶ sim_b_gt
                              │
              ┌───────────────▼───────────────┐
              │   _compute_composite()        │
              │   XOR refusal → -1.0          │
              │   AND refusal → +1.0          │
              │   both substantive → sim_ab   │
              └───────────────────────────────┘
                              │
                    composite_score ∈ [-1, +1]
```
