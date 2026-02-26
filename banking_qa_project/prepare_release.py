import os
from pathlib import Path
from sentence_transformers import SentenceTransformer


def export_for_release():
    base_dir = Path(__file__).resolve().parent
    local_model = base_dir / "models" / "multilingual-e5-large"
    model_name = str(local_model) if local_model.exists() else "intfloat/multilingual-e5-large"
    export_path = base_dir / "release_assets" / "models" / "multilingual-e5-large"

    print(f"Descargando y guardando modelo: {model_name}...")
    model = SentenceTransformer(model_name)
    model.save(str(export_path))

    requirements_path = base_dir / "release_assets" / "requirements.txt"
    with open(requirements_path, "w", encoding="utf-8") as f:
        f.write("sentence-transformers>=2.2.2\n")
        f.write("scikit-learn>=1.0.2\n")
        f.write("numpy>=1.21.2\n")
        f.write("torch>=1.12.0\n")

    print(f"\nListo. Para el release de GitHub, sube el contenido de: {base_dir / 'release_assets'}")
    print("El usuario offline podra cargar el modelo usando la ruta local al directorio.")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    os.makedirs(base_dir / "release_assets" / "models", exist_ok=True)
    export_for_release()
