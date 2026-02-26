import json
import argparse
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Importar la arquitectura del adaptador
class MetricAdapter(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )
    def forward(self, x):
        return self.net(x)


def _print_progress(prefix, current, total):
    pct = (current / total) * 100 if total else 100
    sys.stdout.write(f"\r{prefix}: {current}/{total} ({pct:5.1f}%)")
    if current >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _print_progress_eta(prefix, current, total, start_time):
    elapsed = max(time.time() - start_time, 1e-6)
    rate = current / elapsed if current else 0.0
    remaining = max(total - current, 0)
    eta = (remaining / rate) if rate > 0 else 0.0
    pct = (current / total) * 100 if total else 100
    sys.stdout.write(
        f"\r{prefix}: {current}/{total} ({pct:5.1f}%) | "
        f"elapsed {elapsed:6.1f}s | eta {eta:6.1f}s"
    )
    if current >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def evaluate_adapter(mock=False, mock_samples=10, encode_batch_size=8):
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "separation_dataset.json"
    adapter_path = base_dir / "models" / "metric_adapter.pt"
    local_e5_dir = base_dir / "models" / "multilingual-e5-large"
    model_name = str(local_e5_dir) if local_e5_dir.exists() else "intfloat/multilingual-e5-large"
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if mock:
        sample_count = min(len(data), max(1, mock_samples))
        data = data[:sample_count]
        print(f"[MOCK] Ejecutando con {sample_count} muestras")
    
    print(f"Cargando modelo base: {model_name}")
    st_model = SentenceTransformer(model_name)
    adapter = MetricAdapter(input_dim=1024)
    adapter.load_state_dict(torch.load(adapter_path))
    adapter.eval()
    
    unique_texts = list(set([item["text1"] for item in data] + [item["text2"] for item in data]))
    print("Calentando el modelo (primera inferencia, puede tardar)...")
    warmup_start = time.time()
    st_model.encode(["query: warmup"], show_progress_bar=False, convert_to_numpy=True)
    print(f"Warmup completado en {time.time() - warmup_start:.1f}s")

    print(f"Codificando {len(unique_texts)} textos unicos (batch_size={encode_batch_size})...")
    encode_start = time.time()
    raw_embs = st_model.encode(
        [f"query: {t}" for t in unique_texts],
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"Encoding completado en {time.time() - encode_start:.1f}s")
    
    print("Proyectando embeddings con el adaptador...")
    with torch.no_grad():
        batch_size = 256
        projected_chunks = []
        total = len(raw_embs)
        proj_start = time.time()
        for i in range(0, total, batch_size):
            batch = torch.tensor(raw_embs[i:i + batch_size], dtype=torch.float32)
            projected_chunks.append(adapter(batch).numpy())
            _print_progress_eta("Proyeccion", min(i + batch_size, total), total, proj_start)
        projected_embs = np.vstack(projected_chunks)
    
    text_to_raw = {t: e for t, e in zip(unique_texts, raw_embs)}
    text_to_proj = {t: e for t, e in zip(unique_texts, projected_embs)}
    
    results = {"raw": {}, "projected": {}}
    
    for mode in ["raw", "projected"]:
        print(f"Evaluando distancias ({mode})...")
        mapping = text_to_raw if mode == "raw" else text_to_proj
        distances = {"positive": [], "negative": [], "guardrail_mismatch": []}
        
        total_items = len(data)
        dist_start = time.time()
        for idx, item in enumerate(data, start=1):
            e1 = mapping[item["text1"]]
            e2 = mapping[item["text2"]]
            sim = cosine_similarity([e1], [e2])[0][0]
            distances[item["type"]].append(float(1.0 - sim))
            if idx % 100 == 0 or idx == total_items:
                _print_progress_eta(f"Distancias {mode}", idx, total_items, dist_start)
        
        for t, dists in distances.items():
            if dists:
                results[mode][t] = {
                    "mean": float(np.mean(dists)),
                    "std": float(np.std(dists)),
                    "min": float(np.min(dists)),
                    "max": float(np.max(dists))
                }
            else:
                results[mode][t] = {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None
                }
            
    def _fmt(v):
        return "N/A" if v is None else f"{v:.4f}"

    print(f"{'Tipo':20} | {'Media RAW':10} | {'Media PROYECTADA':15} | {'MEJORA GAP'}")
    print("-" * 70)
    for t in ["positive", "negative", "guardrail_mismatch"]:
        raw_m = results["raw"][t]["mean"]
        proj_m = results["projected"][t]["mean"]
        if raw_m is None or proj_m is None or raw_m == 0:
            gap = "N/A"
        else:
            gap = f"{((proj_m / raw_m) - 1) * 100:+.1f}%"
        print(f"{t:20} | {_fmt(raw_m):10} | {_fmt(proj_m):15} | {gap}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalua metrica raw vs adaptada.")
    parser.add_argument("--mock", action="store_true", help="Ejecuta una prueba rapida con pocas muestras.")
    parser.add_argument("--mock-samples", type=int, default=10, help="Numero de muestras para mock run.")
    parser.add_argument("--encode-batch-size", type=int, default=8, help="Batch size para codificacion.")
    args = parser.parse_args()
    evaluate_adapter(
        mock=args.mock,
        mock_samples=args.mock_samples,
        encode_batch_size=max(1, args.encode_batch_size)
    )
