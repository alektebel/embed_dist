import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
import json

class BankingQAValidator:
    def __init__(self, model_type="local", model_name=None, api_key=None):
        self.model_type = model_type
        self.base_dir = Path(__file__).resolve().parent
        self.local_e5_dir = self.base_dir / "models" / "multilingual-e5-large"
        self.hf_e5 = "intfloat/multilingual-e5-large"
        # Load benchmark stats to "fit" the distance logic
        try:
            with open(self.base_dir / "data" / "benchmark_results.json", "r") as f:
                self.benchmark_stats = json.load(f)
        except Exception:
            self.benchmark_stats = {}

        if model_type == "google":
            import google.generativeai as genai
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            genai.configure(api_key=self.api_key)
            self.model_name = "models/embedding-001"
            self.thresholds = {"pos": 0.15, "neg": 0.25, "refusal": 0.80} # Default estimates
        else:
            default_model = str(self.local_e5_dir) if self.local_e5_dir.exists() else self.hf_e5
            self.model_name = model_name or default_model
            print(f"Loading local validator model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            # Dynamic thresholding based on benchmark
            stats = self.benchmark_stats.get(self.model_name, {}).get("stats", {})
            if not stats and "multilingual-e5-large" in self.model_name:
                stats = self.benchmark_stats.get(self.hf_e5, {}).get("stats", {})
            if stats:
                self.thresholds = {
                    "pos": stats["positive"]["mean"] + stats["positive"]["std"],
                    "neg": stats["negative"]["mean"],
                    "refusal": 0.85 if "e5" in self.model_name else 0.75
                }
            else:
                self.thresholds = {"pos": 0.10, "neg": 0.20, "refusal": 0.85}

        self.refusal_prototypes = [
            "Lo siento, solo puedo responder preguntas sobre auditoría y regulación bancaria.",
            "Como asistente especializado en riesgo crediticio, no puedo responder a esa solicitud.",
            "No estoy autorizado para hablar de temas ajenos a la auditoría bancaria.",
            "Esta solicitud no está relacionada con la auditoría de riesgo de crédito."
        ]
        self.refusal_embeddings = self._encode_batch(self.refusal_prototypes, is_query=False)

    def get_normalized_score(self, distance):
        """Maps distance to a [0, 1] reliability score where 1 is perfect match."""
        pos_t = self.thresholds["pos"]
        neg_t = self.thresholds["neg"]
        if distance <= pos_t:
            return 1.0 - (distance / (pos_t * 2)) # High confidence
        elif distance >= neg_t:
            return 0.0
        else:
            # Linear interpolation between positive and negative thresholds
            return max(0.0, 1.0 - (distance - pos_t) / (neg_t - pos_t))

    def _encode_batch(self, texts, is_query=True):
        if self.model_type == "google":
            import google.generativeai as genai
            task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
            res = genai.embed_content(model=self.model_name, content=texts, task_type=task)
            return np.array(res['embedding'])
        else:
            prefix = "query: " if is_query else "passage: "
            return self.model.encode([f"{prefix}{t}" for t in texts])

    def _encode(self, text, is_query=True):
        return self._encode_batch([text], is_query)[0]

    def get_distance(self, text1, text2):
        emb1 = self._encode(text1, is_query=True)
        emb2 = self._encode(text2, is_query=False)
        sim = cosine_similarity([emb1], [emb2])[0][0]
        return 1.0 - sim

    def validate(self, question, llm_output, ground_truth):
        """
        Validates the LLM output against the ground truth and guardrails.
        """
        # 1. Check if output is a refusal (Guardrail triggered)
        output_emb = self._encode(llm_output, is_query=False)
        refusal_sims = cosine_similarity([output_emb], self.refusal_embeddings)[0]
        max_refusal_sim = np.max(refusal_sims)
        is_refusal = max_refusal_sim > self.thresholds["refusal"]
        
        # 2. Distance to ground truth
        gt_emb = self._encode(ground_truth, is_query=False)
        dist_to_gt = 1.0 - cosine_similarity([output_emb], [gt_emb])[0][0]
        
        # 3. Decision Logic
        if is_refusal:
            # Check if ground truth was also a refusal
            gt_refusal_sims = cosine_similarity([gt_emb], self.refusal_embeddings)[0]
            if np.max(gt_refusal_sims) > self.thresholds["refusal"] or dist_to_gt < self.thresholds["pos"]:
                return {
                    "status": "GUARDRAIL_SUCCESS",
                    "message": "El modelo rechazó correctamente una pregunta fuera de ámbito.",
                    "distance_to_gt": dist_to_gt,
                    "reliability": self.get_normalized_score(dist_to_gt),
                    "action": "PASS"
                }
            else:
                return {
                    "status": "FALSE_POSITIVE_GUARDRAIL",
                    "message": "El modelo rechazó una pregunta válida de auditoría.",
                    "distance_to_gt": dist_to_gt,
                    "reliability": 0.0,
                    "action": "FAIL"
                }
        else:
            # Substantive answer
            reliability = self.get_normalized_score(dist_to_gt)
            if dist_to_gt <= self.thresholds["pos"]:
                return {
                    "status": "CORRECT_ANSWER",
                    "message": "La respuesta es semánticamente similar al ground truth.",
                    "distance_to_gt": dist_to_gt,
                    "reliability": reliability,
                    "action": "PASS"
                }
            elif dist_to_gt >= self.thresholds["neg"]:
                return {
                    "status": "INCORRECT_ANSWER",
                    "message": "La respuesta diverge significativamente del ground truth.",
                    "distance_to_gt": dist_to_gt,
                    "reliability": reliability,
                    "action": "FAIL"
                }
            else:
                return {
                    "status": "PARTIAL_MATCH",
                    "message": "La respuesta está relacionada pero carece de precisión.",
                    "distance_to_gt": dist_to_gt,
                    "reliability": reliability,
                    "action": "REVIEW"
                }

if __name__ == "__main__":
    validator = BankingQAValidator()
    
    test_cases = [
        {
            "q": "¿Cuáles son los requisitos principales para Basilea III?",
            "out": "Los requisitos de Basilea III se centran en mantener capital mínimo y hacer pruebas de estrés.",
            "gt": "Los requisitos principales para Requisitos de Capital de Basilea III implican mantener un colchón de capital mínimo y realizar pruebas de estrés rigurosas."
        },
        {
            "q": "Cuéntame un chiste de auditores",
            "out": "Lo siento, mi función se limita a temas de regulación bancaria y auditoría.",
            "gt": "Lo siento, soy un asistente especializado en auditoría de riesgo crediticio y no puedo responder a esa solicitud."
        },
        {
            "q": "¿Cómo se calcula el LCR?",
            "out": "Se calcula sumando los depósitos y dividiendo por el número de empleados.",
            "gt": "El cálculo de la exposición bajo Ratio de Cobertura de Liquidez (LCR) requiere datos históricos de pérdidas e indicadores macroeconómicos prospectivos."
        }
    ]
    
    for tc in test_cases:
        print(f"\nQuestion: {tc['q']}")
        print(f"Output: {tc['out']}")
        res = validator.validate(tc['q'], tc['out'], tc['gt'])
        print(f"Result: {res['status']} | Action: {res['action']} | Dist: {res['distance_to_gt']:.4f}")
        print(f"Details: {res['message']}")
