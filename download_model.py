"""
download_model.py
-----------------
Run this ONCE on an internet-connected machine.
It downloads the sentence-transformer model weights from HuggingFace
and saves them to ./models/ so every other script works fully offline.

Usage:
    python download_model.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download.log"),
    ],
)
log = logging.getLogger("download_model")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAVE_DIR = Path("models") / "all-MiniLM-L6-v2"


def main() -> None:
    log.info("Starting model download: %s", MODEL_NAME)
    log.info("Target directory: %s", SAVE_DIR.resolve())

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Save directory ready.")

    log.info("Fetching model from HuggingFace (this may take a minute)…")
    model = SentenceTransformer(MODEL_NAME)

    log.info("Saving model to disk at %s …", SAVE_DIR)
    model.save(str(SAVE_DIR))
    log.info("Model saved successfully.")

    # Quick sanity check
    log.info("Running offline sanity check with the saved model…")
    model_offline = SentenceTransformer(str(SAVE_DIR))
    emb = model_offline.encode(["hello world"])
    log.info("Sanity check passed — embedding shape: %s", emb.shape)
    log.info("All done. You can now transfer this directory to an offline machine.")


if __name__ == "__main__":
    main()
