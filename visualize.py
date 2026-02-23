"""
visualize.py
------------
Embed a list of phrases, reduce to 2-D with PCA *and* t-SNE,
and produce scatter plots showing which phrases are semantically close.

Usage:
    # Use built-in sample phrases
    python visualize.py

    # Load phrases from a file (one per line, optional "label::phrase" format)
    python visualize.py --file phrases.txt

    # Save plots instead of showing them interactively
    python visualize.py --save

Output files (when --save):
    embed_pca.png
    embed_tsne.png
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("visualize.log"),
    ],
)
log = logging.getLogger("visualize")

MODEL_DIR = Path("models") / "all-MiniLM-L6-v2"

# ── Sample phrases grouped by topic ──────────────────────────────────────────
SAMPLE_GROUPS: dict[str, list[str]] = {
    "AI / ML": [
        "machine learning",
        "deep learning",
        "neural network",
        "artificial intelligence",
        "transformer model",
        "gradient descent",
        "backpropagation",
        "computer vision",
        "natural language processing",
        "reinforcement learning",
    ],
    "Food": [
        "pepperoni pizza",
        "margherita pizza",
        "cheeseburger",
        "veggie burger",
        "spaghetti carbonara",
        "pad thai noodles",
        "sushi roll",
        "fish and chips",
        "beef tacos",
        "chicken curry",
    ],
    "Sports": [
        "football match",
        "soccer game",
        "basketball dunk",
        "tennis serve",
        "swimming race",
        "marathon running",
        "cycling tour",
        "baseball home run",
        "volleyball spike",
        "ice hockey goal",
    ],
    "Finance": [
        "stock market crash",
        "bond yield curve",
        "inflation rate",
        "interest rates rise",
        "cryptocurrency Bitcoin",
        "hedge fund strategy",
        "IPO listing",
        "dividend payout",
        "venture capital investment",
        "central bank policy",
    ],
}


def load_model():
    log.info("Loading model from %s …", MODEL_DIR.resolve())
    if not MODEL_DIR.exists():
        log.error("Model not found. Run `python download_model.py` first.")
        sys.exit(1)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install -r requirements.txt")
        sys.exit(1)
    model = SentenceTransformer(str(MODEL_DIR))
    log.info("Model ready. Embedding dim: %d", model.get_sentence_embedding_dimension())
    return model


def load_phrases_from_file(path: str) -> dict[str, list[str]]:
    """
    Accept either:
      - plain lines  →  all grouped under "Custom"
      - "Label::phrase" lines  →  grouped by label
    """
    lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    log.info("Read %d non-empty lines from %s", len(lines), path)

    groups: dict[str, list[str]] = {}
    for line in lines:
        if "::" in line:
            label, phrase = line.split("::", 1)
            groups.setdefault(label.strip(), []).append(phrase.strip())
        else:
            groups.setdefault("Custom", []).append(line)

    for label, phrases in groups.items():
        log.info("  Group %r: %d phrases", label, len(phrases))
    return groups


def embed_groups(model, groups: dict[str, list[str]]) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (embeddings, labels, phrases) arrays aligned by index."""
    labels, phrases = [], []
    for label, group_phrases in groups.items():
        for phrase in group_phrases:
            labels.append(label)
            phrases.append(phrase)

    log.info("Encoding %d phrases…", len(phrases))
    embeddings = model.encode(phrases, show_progress_bar=True, normalize_embeddings=True)
    log.info("Embeddings shape: %s", embeddings.shape)
    return embeddings, labels, phrases


def _assign_colors(labels: list[str]) -> tuple[list, dict]:
    unique = list(dict.fromkeys(labels))   # preserve insertion order
    palette = cm.get_cmap("tab10", len(unique))
    color_map = {label: palette(i) for i, label in enumerate(unique)}
    colors = [color_map[l] for l in labels]
    log.info("Assigned %d distinct colors for groups: %s", len(unique), unique)
    return colors, color_map


def plot_reduction(
    coords: np.ndarray,
    labels: list[str],
    phrases: list[str],
    color_map: dict,
    title: str,
    save_path: str | None = None,
) -> None:
    log.info("Plotting %s…", title)
    fig, ax = plt.subplots(figsize=(13, 9))

    unique_labels = list(color_map.keys())
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c=[color_map[label]],
            label=label,
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    # Annotate each point
    for i, phrase in enumerate(phrases):
        ax.annotate(
            phrase,
            (coords[i, 0], coords[i, 1]),
            fontsize=7.5,
            alpha=0.9,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        log.info("Saved plot → %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def run_pca(embeddings: np.ndarray) -> np.ndarray:
    log.info("Running PCA to 2-D…")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    log.info(
        "PCA explained variance: PC1=%.2f%%  PC2=%.2f%%  (total=%.2f%%)",
        pca.explained_variance_ratio_[0] * 100,
        pca.explained_variance_ratio_[1] * 100,
        pca.explained_variance_ratio_.sum() * 100,
    )
    return coords


def run_tsne(embeddings: np.ndarray, perplexity: float = 5.0) -> np.ndarray:
    n = len(embeddings)
    perp = min(perplexity, max(2.0, n / 3 - 1))
    log.info("Running t-SNE to 2-D (perplexity=%.1f, n=%d)…", perp, n)
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        n_iter=1000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)
    log.info("t-SNE done. KL divergence: %.4f", tsne.kl_divergence_)
    return coords


def log_nearest_neighbors(embeddings: np.ndarray, phrases: list[str], top_k: int = 3) -> None:
    """Log the top-k nearest neighbors for every phrase."""
    log.info("── Nearest neighbors (top %d) ──", top_k)
    # embeddings already L2-normalised; dot product == cosine similarity
    sim_matrix = embeddings @ embeddings.T
    for i, phrase in enumerate(phrases):
        row = sim_matrix[i].copy()
        row[i] = -999  # exclude self
        top_idx = np.argsort(row)[::-1][:top_k]
        neighbors = [(phrases[j], float(row[j])) for j in top_idx]
        neighbor_str = ", ".join(f"{p!r} ({s:.3f})" for p, s in neighbors)
        log.info("  %r  →  %s", phrase, neighbor_str)


def parse_args():
    p = argparse.ArgumentParser(description="Visualize phrase embedding distances.")
    p.add_argument("--file", "-f", help="Text file with phrases (one per line).")
    p.add_argument("--save", "-s", action="store_true", help="Save plots as PNG files.")
    p.add_argument("--no-tsne", action="store_true", help="Skip t-SNE (faster).")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    model = load_model()

    if args.file:
        groups = load_phrases_from_file(args.file)
    else:
        log.info("No --file provided. Using built-in sample phrases.")
        groups = SAMPLE_GROUPS

    embeddings, labels, phrases = embed_groups(model, groups)
    colors, color_map = _assign_colors(labels)

    log_nearest_neighbors(embeddings, phrases)

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca_coords = run_pca(embeddings)
    plot_reduction(
        pca_coords,
        labels,
        phrases,
        color_map,
        title="Phrase Embeddings — PCA (2-D)",
        save_path="embed_pca.png" if args.save else None,
    )

    # ── t-SNE ────────────────────────────────────────────────────────────────
    if not args.no_tsne:
        tsne_coords = run_tsne(embeddings)
        plot_reduction(
            tsne_coords,
            labels,
            phrases,
            color_map,
            title="Phrase Embeddings — t-SNE (2-D)",
            save_path="embed_tsne.png" if args.save else None,
        )

    log.info("Visualization complete.")


if __name__ == "__main__":
    main()
