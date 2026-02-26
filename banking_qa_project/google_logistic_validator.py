import json
import numpy as np
import google.generativeai as genai
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import time
import os
from pathlib import Path

class GoogleLogisticValidator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found.")
        genai.configure(api_key=self.api_key)
        self.model_name = "models/embedding-001"
        self.clf = LogisticRegression(max_iter=1000)
        self.base_dir = Path(__file__).resolve().parent

    def get_embeddings(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        result = genai.embed_content(model=self.model_name, content=texts, task_type=task_type)
        return np.array(result['embedding'])

    def prepare_features(self, emb1, emb2):
        """
        Combine two embeddings into a single feature vector for the classifier.
        Options: Concatenation, Absolute Difference, or Element-wise Product.
        Difference + Product is usually very effective for similarity tasks.
        """
        diff = np.abs(emb1 - emb2)
        prod = emb1 * emb2
        return np.hstack((diff, prod))

    def train_on_dataset(self, dataset_path):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        unique_texts = list(set([item["text1"] for item in dataset] + [item["text2"] for item in dataset]))
        print(f"Obteniendo embeddings de Google para {len(unique_texts)} textos...")
        
        text_to_emb = {}
        batch_size = 100
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i:i+batch_size]
            embs = self.get_embeddings(batch)
            for text, emb in zip(batch, embs):
                text_to_emb[text] = emb
            time.sleep(1)

        X = []
        y = []
        for item in dataset:
            feat = self.prepare_features(text_to_emb[item["text1"]], text_to_emb[item["text2"]])
            X.append(feat)
            # Label 1 for positive (correct), 0 for negative/guardrail
            y.append(1 if item["type"] == "positive" else 0)

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Entrenando Regresión Logística...")
        self.clf.fit(X_train, y_train)
        
        preds = self.clf.predict(X_test)
        print("\nReporte de Clasificación (Google API + Logistic Regression):")
        print(classification_report(y_test, preds, target_names=["Incorrecto/Guardrail", "Correcto"]))
        
        # Guardar el modelo
        model_save_path = self.base_dir / "models" / "google_lr_validator.joblib"
        os.makedirs(model_save_path.parent, exist_ok=True)
        joblib.dump(self.clf, model_save_path)
        print(f"Modelo guardado en {model_save_path}")

    def validate(self, question, llm_output, ground_truth):
        """
        Usa el modelo entrenado para dar una probabilidad de acierto.
        """
        # Obtenemos embeddings (esto requiere API KEY)
        embs = self.get_embeddings([llm_output, ground_truth])
        feat = self.prepare_features(embs[0], embs[1]).reshape(1, -1)
        
        prob = self.clf.predict_proba(feat)[0][1] # Probabilidad de clase 'Correcto'
        
        status = "PASS" if prob > 0.8 else "FAIL" if prob < 0.2 else "REVIEW"
        return {
            "probability_correct": round(float(prob), 4),
            "status": status,
            "method": "Google API + Logistic Regression"
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python google_logistic_validator.py <API_KEY>")
        sys.exit(1)
        
    validator = GoogleLogisticValidator(api_key=sys.argv[1])
    validator.train_on_dataset(validator.base_dir / "data" / "separation_dataset.json")
