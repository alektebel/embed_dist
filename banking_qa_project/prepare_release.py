import os
from sentence_transformers import SentenceTransformer
import shutil

def export_for_release():
    model_name = "intfloat/multilingual-e5-large"
    export_path = "banking_qa_project/release_assets/models/multilingual-e5-large"
    
    print(f"Descargando y guardando modelo: {model_name}...")
    model = SentenceTransformer(model_name)
    model.save(export_path)
    
    # Crear un archivo de requerimientos específico
    with open("banking_qa_project/release_assets/requirements.txt", "w") as f:
        f.write("sentence-transformers>=2.2.2\n")
        f.write("scikit-learn>=1.0.2\n")
        f.write("numpy>=1.21.2\n")
        f.write("torch>=1.12.0\n")

    print(f"\nListo. Para el release de GitHub, sube el contenido de: banking_qa_project/release_assets/")
    print("El usuario offline podrá cargar el modelo usando la ruta local al directorio.")

if __name__ == "__main__":
    os.makedirs("banking_qa_project/release_assets/models", exist_ok=True)
    export_for_release()
