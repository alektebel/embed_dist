"""
distance.py
-----------
Compute the cosine similarity (and distance) between any two strings
using a locally-saved sentence-transformer model.

Usage:
    # Interactive
    python distance.py

    # Direct arguments
    python distance.py "machine learning" "deep learning"

    # Batch: compare one query against many targets from a file
    python distance.py --query "neural network" --file phrases.txt
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("distance.log"),
    ],
)
log = logging.getLogger("distance")

MODEL_DIR = Path("models") / "all-MiniLM-L6-v2"


def load_model():
    log.info("Loading model from %s …", MODEL_DIR.resolve())
    if not MODEL_DIR.exists():
        log.error(
            "Model directory not found. Run `python download_model.py` first."
        )
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    model = SentenceTransformer(str(MODEL_DIR))
    log.info("Model loaded. Embedding dimension: %d", model.get_sentence_embedding_dimension())
    return model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [-1, 1]. Higher = more similar."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        log.warning("One of the embeddings has zero norm — returning 0.0")
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compare(model, text_a: str, text_b: str) -> dict:
    log.info("Encoding  A: %r", text_a)
    log.info("Encoding  B: %r", text_b)

    emb_a, emb_b = model.encode([text_a, text_b])
    log.debug("Embedding A shape: %s  norm: %.4f", emb_a.shape, np.linalg.norm(emb_a))
    log.debug("Embedding B shape: %s  norm: %.4f", emb_b.shape, np.linalg.norm(emb_b))

    sim = cosine_similarity(emb_a, emb_b)
    dist = 1.0 - sim          # cosine distance in [0, 2]
    euclid = float(np.linalg.norm(emb_a - emb_b))

    log.info("─" * 50)
    log.info("Cosine similarity : %+.6f  (1 = identical, 0 = unrelated, -1 = opposite)", sim)
    log.info("Cosine distance   :  %.6f  (0 = identical, 2 = maximally distant)", dist)
    log.info("Euclidean distance:  %.6f", euclid)
    log.info("─" * 50)

    return {"similarity": sim, "cosine_distance": dist, "euclidean_distance": euclid}


def batch_compare(model, query: str, targets: list[str]) -> list[dict]:
    log.info("Batch mode — query: %r against %d targets", query, len(targets))
    all_texts = [query] + targets
    embeddings = model.encode(all_texts, show_progress_bar=True)
    emb_q = embeddings[0]

    results = []
    for i, target in enumerate(targets):
        emb_t = embeddings[i + 1]
        sim = cosine_similarity(emb_q, emb_t)
        dist = 1.0 - sim
        results.append({"text": target, "similarity": sim, "cosine_distance": dist})
        log.info("  [%3d] sim=%.4f dist=%.4f  %r", i + 1, sim, dist, target)

    results.sort(key=lambda r: r["similarity"], reverse=True)
    log.info("Top result: %r (similarity=%.4f)", results[0]["text"], results[0]["similarity"])
    return results


def interactive(model) -> None:
    print("\nEnter two phrases to compare (Ctrl-C to quit).\n")
    while True:
        try:
            a = input("Phrase A: ").strip()
            b = input("Phrase B: ").strip()
            if not a or not b:
                log.warning("Empty input — skipping.")
                continue
            compare(model, a, b)
            print()
        except KeyboardInterrupt:
            print("\nBye.")
            break


def parse_args():
    p = argparse.ArgumentParser(description="Compute embedding distance between strings.")
    p.add_argument("positional", nargs="*", help="Two strings to compare directly.")
    p.add_argument("--query", "-q", help="Query string for batch mode.")
    p.add_argument("--file", "-f", help="File with one phrase per line (batch mode).")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Debug logging enabled.")

    model = load_model()

    if args.query and args.file:
        # Batch mode: query vs file
        file_path = Path(args.file)
        if not file_path.exists():
            log.error("File not found: %s", args.file)
            sys.exit(1)
        targets = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]
        log.info("Read %d phrases from %s", len(targets), args.file)
        batch_compare(model, args.query, targets)

    elif len(args.positional) == 2:
        # Direct two-string comparison
        compare(model, args.positional[0], args.positional[1])

    else:
        # Interactive mode
        interactive(model)


if __name__ == "__main__":
    main()
