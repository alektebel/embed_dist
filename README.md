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
├── download_model.py   # One-time online setup
├── distance.py         # Pairwise / batch distance tool
├── visualize.py        # Cluster scatter plots
├── phrases.txt         # Sample phrases file
├── requirements.txt
├── models/             # Model weights saved here
│   └── all-MiniLM-L6-v2/
└── README.md
```
