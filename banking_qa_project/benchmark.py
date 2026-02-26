import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import os
from pathlib import Path

# Fix path resolution
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODELS = [
    "BAAI/bge-large-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/multilingual-e5-large"
]

def load_dataset(path):
    if not os.path.isabs(path):
        path = DATA_DIR / os.path.basename(path)
    with open(path, "r") as f:
        return json.load(f)

def benchmark_models(dataset_path):
    dataset = load_dataset(dataset_path)
    
    # Collect all unique strings
    unique_texts = set()
    for item in dataset:
        unique_texts.add(item["text1"])
        unique_texts.add(item["text2"])
    
    unique_texts = list(unique_texts)
    results = {}

    for model_name in MODELS:
        print(f"\n--- Benchmarking {model_name} ---")
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        # Batch encode
        print(f"Encoding {len(unique_texts)} unique strings...")
        
        # Handle E5 prefix
        encoding_texts = unique_texts
        if "e5" in model_name.lower():
            encoding_texts = [f"passage: {t}" for t in unique_texts]
            
        embeddings = model.encode(encoding_texts, convert_to_numpy=True, show_progress_bar=True)
        text_to_emb = {text: emb for text, emb in zip(unique_texts, embeddings)}
        
        distances = []
        labels = []
        
        for item in dataset:
            emb1 = text_to_emb[item["text1"]]
            emb2 = text_to_emb[item["text2"]]
            
            # Use dot product for BGE if needed, but cosine is standard for distance comparison
            # Normalize for cosine similarity
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            sim = np.dot(emb1, emb2) / (norm1 * norm2)
            dist = 1.0 - sim
            
            distances.append(float(dist))
            labels.append(item["type"])
            
        # Analyze separation
        stats = {}
        for t in set(labels):
            dists = [distances[i] for i, l in enumerate(labels) if l == t]
            stats[t] = {
                "mean": np.mean(dists),
                "std": np.std(dists),
                "min": np.min(dists),
                "max": np.max(dists)
            }
            
        results[model_name] = {
            "stats": stats
        }
        
    return results

if __name__ == "__main__":
    # Ensure separation dataset exists
    sep_dataset_path = DATA_DIR / "separation_dataset.json"
    if not sep_dataset_path.exists():
        print(f"Error: {sep_dataset_path} not found. Running generate_data.py first...")
        import subprocess
        subprocess.run(["python", str(BASE_DIR / "generate_data.py")], check=True)

    benchmark_results = benchmark_models(str(sep_dataset_path))
    
    output_path = DATA_DIR / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(benchmark_results, f, indent=4)
        
    for model, data in benchmark_results.items():
        print(f"\nModel: {model}")
        print(f"{'Type':25} | {'Mean Dist':10} | {'Std':10} | {'Range':20}")
        print("-" * 75)
        for label, s in data["stats"].items():
            print(f"{label:25} | {s['mean']:.4f}     | {s['std']:.4f} | {s['min']:.3f} to {s['max']:.3f}")
