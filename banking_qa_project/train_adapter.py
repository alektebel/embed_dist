import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

class DistanceDataset(Dataset):
    def __init__(self, data, model):
        self.samples = []
        unique_texts = list(set([item["text1"] for item in data] + [item["text2"] for item in data]))
        print(f"Pre-calculando embeddings para {len(unique_texts)} textos...")
        
        # Usar prefijos de E5 para el entrenamiento
        text_to_emb = {}
        batch_size = 32
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i:i+batch_size]
            # Mezclamos query y passage para que el adaptador sea robusto
            embs = model.encode([f"query: {t}" for t in batch])
            for t, e in zip(batch, embs):
                text_to_emb[t] = e
                
        for item in data:
            self.samples.append({
                "emb1": text_to_emb[item["text1"]],
                "emb2": text_to_emb[item["text2"]],
                "label": 1.0 if item["type"] == "positive" else 0.0
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.tensor(s["emb1"]), torch.tensor(s["emb2"]), torch.tensor(s["label"])

class MetricAdapter(nn.Module):
    def __init__(self, input_dim=1024): # 1024 para e5-large
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

def train_adapter():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "separation_dataset.json"
    with open(data_path, "r") as f:
        data = json.load(f)

    st_model = SentenceTransformer("intfloat/multilingual-e5-large")
    dataset = DistanceDataset(data, st_model)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    adapter = MetricAdapter(input_dim=1024)
    optimizer = optim.Adam(adapter.parameters(), lr=1e-4)
    criterion = nn.CosineEmbeddingLoss(margin=0.3)

    print("Iniciando entrenamiento del adaptador...")
    adapter.train()
    for epoch in range(50):
        total_loss = 0
        for emb1, emb2, target in loader:
            # target=1 para similares, target=-1 para disímiles
            # En nuestro dataset target es 1 (pos) o 0 (neg). Convertimos 0 a -1.
            t = (target * 2 - 1)
            
            optimizer.zero_grad()
            out1 = adapter(emb1)
            out2 = adapter(emb2)
            
            loss = criterion(out1, out2, t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 | Loss: {total_loss/len(loader):.4f}")

    # Guardar adaptador
    save_path = base_dir / "models" / "metric_adapter.pt"
    os.makedirs(save_path.parent, exist_ok=True)
    torch.save(adapter.state_dict(), save_path)
    print(f"Adaptador guardado en {save_path}")

if __name__ == "__main__":
    import os
    train_adapter()
