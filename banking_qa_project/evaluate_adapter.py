import json
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

def evaluate_adapter():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "separation_dataset.json"
    adapter_path = base_dir / "models" / "metric_adapter.pt"
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    st_model = SentenceTransformer("intfloat/multilingual-e5-large")
    adapter = MetricAdapter(input_dim=1024)
    adapter.load_state_dict(torch.load(adapter_path))
    adapter.eval()
    
    unique_texts = list(set([item["text1"] for item in data] + [item["text2"] for item in data]))
    raw_embs = st_model.encode([f"query: {t}" for t in unique_texts])
    
    with torch.no_grad():
        projected_embs = adapter(torch.tensor(raw_embs)).numpy()
    
    text_to_raw = {t: e for t, e in zip(unique_texts, raw_embs)}
    text_to_proj = {t: e for t, e in zip(unique_texts, projected_embs)}
    
    results = {"raw": {}, "projected": {}}
    
    for mode in ["raw", "projected"]:
        mapping = text_to_raw if mode == "raw" else text_to_proj
        distances = {"positive": [], "negative": [], "guardrail_mismatch": []}
        
        for item in data:
            e1 = mapping[item["text1"]]
            e2 = mapping[item["text2"]]
            sim = cosine_similarity([e1], [e2])[0][0]
            distances[item["type"]].append(float(1.0 - sim))
        
        for t, dists in distances.items():
            results[mode][t] = {
                "mean": np.mean(dists),
                "std": np.std(dists),
                "min": np.min(dists),
                "max": np.max(dists)
            }
            
    print(f"{'Tipo':20} | {'Media RAW':10} | {'Media PROYECTADA':15} | {'MEJORA GAP'}")
    print("-" * 70)
    for t in ["positive", "negative", "guardrail_mismatch"]:
        raw_m = results["raw"][t]["mean"]
        proj_m = results["projected"][t]["mean"]
        print(f"{t:20} | {raw_m:.4f}     | {proj_m:.4f}           | {((proj_m/raw_m)-1)*100:+.1f}%")

if __name__ == "__main__":
    evaluate_adapter()
