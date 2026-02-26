import json
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from pathlib import Path

class GoogleEmbeddingsEvaluator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment or passed as argument.")
        genai.configure(api_key=self.api_key)
        self.model_name = "models/embedding-001"

    def get_embeddings(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        """Batch get embeddings from Google API."""
        result = genai.embed_content(
            model=self.model_name,
            content=texts,
            task_type=task_type
        )
        return np.array(result['embedding'])

    def evaluate_separation(self, dataset_path):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        unique_texts = list(set([item["text1"] for item in dataset] + [item["text2"] for item in dataset]))
        
        print(f"Requesting embeddings for {len(unique_texts)} texts from Google API...")
        # Split into batches of 100 to avoid API limits if necessary
        text_to_emb = {}
        batch_size = 100
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i:i+batch_size]
            embs = self.get_embeddings(batch)
            for text, emb in zip(batch, embs):
                text_to_emb[text] = emb
            time.sleep(1) # Rate limiting caution

        distances = []
        labels = []
        for item in dataset:
            emb1 = text_to_emb[item["text1"]]
            emb2 = text_to_emb[item["text2"]]
            
            sim = cosine_similarity([emb1], [emb2])[0][0]
            distances.append(float(1.0 - sim))
            labels.append(item["type"])

        stats = {}
        for t in set(labels):
            dists = [distances[i] for i, l in enumerate(labels) if l == t]
            stats[t] = {
                "mean": np.mean(dists),
                "std": np.std(dists),
                "min": np.min(dists),
                "max": np.max(dists)
            }
        return stats

if __name__ == "__main__":
    # Example usage (will fail without API key)
    import sys
    if len(sys.argv) < 2:
        print("Usage: python google_eval.py <API_KEY>")
        sys.exit(1)
    
    evaluator = GoogleEmbeddingsEvaluator(api_key=sys.argv[1])
    base_dir = Path(__file__).resolve().parent
    stats = evaluator.evaluate_separation(base_dir / "data" / "separation_dataset.json")
    
    print("\nGoogle embedding-001 Results:")
    for label, s in stats.items():
        print(f"{label:25} | Mean Dist: {s['mean']:.4f} | Std: {s['std']:.4f}")
