"""
train_refusal_classifier.py
---------------------------
Fine-tunes a lightweight MLP head on top of frozen all-MiniLM-L6-v2 embeddings
to classify text as either a refusal/guardrail response (label 0) or a
substantive answer (label 1).

The backbone is never updated — only the 384→64→1 head is trained.  Training
data lives in ``data/refusal_training.jsonl`` (one JSON object per line with
fields ``text`` and ``label``).

Usage::

    python train_refusal_classifier.py
    python train_refusal_classifier.py --data data/refusal_training.jsonl \\
                                        --output models/refusal_classifier.pt \\
                                        --epochs 50 --lr 1e-3 -v

Design decisions
~~~~~~~~~~~~~~~~
* Embeddings are pre-computed once and cached as tensors — the sentence
  transformer is never called during the training loop, keeping the loop fast.
* Early stopping on validation loss (patience=5) prevents overfitting on the
  small dataset.
* A fixed random seed (42) makes every run fully reproducible.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)
log = logging.getLogger("train_refusal_classifier")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
EMBED_DIM = 384
DEFAULT_DATA = Path("data/refusal_training.jsonl")
DEFAULT_OUTPUT = Path("models/refusal_classifier.pt")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class RefusalClassifier(nn.Module):
    """Binary MLP classifier that predicts whether text is a refusal.

    Args:
        embed_dim: Dimensionality of the input sentence embeddings (default 384
            for all-MiniLM-L6-v2).

    Example::

        model = RefusalClassifier()
        x = torch.randn(8, 384)
        probs = model(x)          # shape (8,), values in [0, 1]
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape ``(batch, embed_dim)``.

        Returns:
            Float tensor of shape ``(batch,)`` with values in ``[0, 1]``.
            Values closer to 0 indicate a refusal; closer to 1 indicate a
            substantive response.
        """
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> tuple[list[str], list[int]]:
    """Read a JSONL file with ``text`` and ``label`` fields.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        A tuple ``(texts, labels)`` where ``labels`` contains 0 (refusal) or
        1 (substantive).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is missing ``text`` or ``label`` fields.

    Example::

        texts, labels = load_dataset(Path("data/refusal_training.jsonl"))
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    texts, labels = [], []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {i}: invalid JSON — {exc}") from exc
        if "text" not in obj or "label" not in obj:
            raise ValueError(f"Line {i}: missing 'text' or 'label' field")
        texts.append(obj["text"])
        labels.append(int(obj["label"]))

    log.info("Loaded %d examples (%d refusal, %d substantive)",
             len(texts),
             labels.count(0),
             labels.count(1))
    return texts, labels


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode texts using the local all-MiniLM-L6-v2 model.

    The backbone is imported via ``distance.load_model()`` so the same model
    directory resolution logic applies.

    Args:
        texts: List of strings to encode.

    Returns:
        Float32 numpy array of shape ``(len(texts), 384)``.

    Example::

        embeddings = embed_texts(["Hello world", "Goodbye world"])
        # embeddings.shape == (2, 384)
    """
    # Import here to avoid circular issues; distance.py lives in the same dir.
    from distance import load_model  # noqa: PLC0415

    model = load_model()
    log.info("Encoding %d texts …", len(texts))
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              convert_to_numpy=True)
    log.info("Embeddings shape: %s", embeddings.shape)
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_indices(n: int, train_frac: float = 0.8, val_frac: float = 0.1,
                   seed: int = SEED) -> tuple[list[int], list[int], list[int]]:
    """Shuffle and split indices into train / val / test.

    Args:
        n: Total number of samples.
        train_frac: Fraction for training set.
        val_frac: Fraction for validation set.
        seed: Random seed.

    Returns:
        Tuple of ``(train_idx, val_idx, test_idx)`` lists.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n).tolist()
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def _make_loader(X: torch.Tensor, y: torch.Tensor,
                 batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(X, y), batch_size=batch_size,
                      shuffle=shuffle)


def train_epoch(model: RefusalClassifier,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module) -> float:
    """Run one training epoch and return mean loss.

    Args:
        model: The classifier.
        loader: Training data loader.
        optimizer: Torch optimiser.
        criterion: Loss function (BCELoss).

    Returns:
        Mean loss over all batches.
    """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: RefusalClassifier,
             loader: DataLoader,
             criterion: nn.Module,
             threshold: float = 0.5) -> dict:
    """Evaluate model on a data loader.

    Args:
        model: The classifier.
        loader: Data loader to evaluate on.
        criterion: Loss function.
        threshold: Decision boundary for binary prediction.

    Returns:
        Dict with keys ``loss``, ``accuracy``, ``f1``, ``tp``, ``tn``,
        ``fp``, ``fn``.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        probs = model(X_batch)
        loss = criterion(probs, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = (probs >= threshold).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    tp = int(((all_preds == 1) & (all_labels == 1)).sum())
    tn = int(((all_preds == 0) & (all_labels == 0)).sum())
    fp = int(((all_preds == 1) & (all_labels == 0)).sum())
    fn = int(((all_preds == 0) & (all_labels == 1)).sum())
    accuracy = (tp + tn) / max(len(all_labels), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "accuracy": accuracy,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(data_path: Path = DEFAULT_DATA,
          output_path: Path = DEFAULT_OUTPUT,
          threshold: float = 0.5,
          epochs: int = 50,
          lr: float = 1e-3,
          batch_size: int = 32,
          patience: int = 5) -> RefusalClassifier:
    """Full training pipeline for the refusal classifier.

    Loads the dataset, encodes all texts with frozen embeddings, trains the
    MLP head with early stopping, evaluates on the test set, and saves the
    model weights.

    Args:
        data_path: Path to the JSONL training dataset.
        output_path: Destination for the saved model state dict.
        threshold: Decision boundary used during evaluation.
        epochs: Maximum number of training epochs.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        patience: Early-stopping patience in epochs.

    Returns:
        The trained ``RefusalClassifier`` in eval mode.

    Raises:
        FileNotFoundError: If ``data_path`` does not exist.
    """
    _set_seed(SEED)

    # 1. Load & encode
    texts, labels = load_dataset(data_path)
    embeddings = embed_texts(texts)

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    # 2. Split
    train_idx, val_idx, test_idx = _split_indices(len(texts))
    log.info("Split — train: %d  val: %d  test: %d",
             len(train_idx), len(val_idx), len(test_idx))

    train_loader = _make_loader(X[train_idx], y[train_idx], batch_size, shuffle=True)
    val_loader = _make_loader(X[val_idx], y[val_idx], batch_size, shuffle=False)
    test_loader = _make_loader(X[test_idx], y[test_idx], batch_size, shuffle=False)

    # 3. Model, loss, optimiser
    model = RefusalClassifier(embed_dim=EMBED_DIM)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # 4. Training loop with early stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, criterion, threshold)
        val_loss = val_metrics["loss"]

        log.info("Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                 epoch, epochs, train_loss, val_loss, val_metrics["accuracy"])

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info("Early stopping triggered at epoch %d (patience=%d)",
                         epoch, patience)
                break

    # 5. Restore best weights and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_metrics = evaluate(model, test_loader, criterion, threshold)
    log.info("─" * 50)
    log.info("Test accuracy : %.1f%%", test_metrics["accuracy"] * 100)
    log.info("Test F1 score : %.3f", test_metrics["f1"])
    log.info("Confusion matrix:")
    log.info("       Pred 0  Pred 1")
    log.info("True 0   %2d      %2d", test_metrics["tn"], test_metrics["fp"])
    log.info("True 1   %2d      %2d", test_metrics["fn"], test_metrics["tp"])
    log.info("─" * 50)

    # 6. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    log.info("Model saved → %s", output_path)

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser(
        description="Train a refusal classifier on top of frozen embeddings."
    )
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="Path to JSONL training dataset (default: %(default)s)")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Output path for model weights (default: %(default)s)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for evaluation (default: %(default)s)")
    p.add_argument("--epochs", type=int, default=50,
                   help="Maximum training epochs (default: %(default)s)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate (default: %(default)s)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG logging")
    return p.parse_args()


def main() -> None:
    """Entry point for the training script."""
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Debug logging enabled.")

    train(
        data_path=Path(args.data),
        output_path=Path(args.output),
        threshold=args.threshold,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
