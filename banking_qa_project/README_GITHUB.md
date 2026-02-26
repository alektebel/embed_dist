# Banking QA Validation Project

## Model Sizes
- **Base Model (multilingual-e5-large)**: ~2.2 GB
- **Metric Adapter**: ~3.1 MB
- **Dataset**: < 1 MB

## GitHub Upload Constraints
GitHub limits individual files to **100MB**. 
1. The **Code** and **Metric Adapter** are small and will be pushed to the repository.
2. The **Base Model (2.2GB)** exceeds the limit. It should be handled via **Git LFS** or downloaded manually.

## Hybrid Approaches
- **Google API + Logistic Regression**: We've added `google_logistic_validator.py`. This uses Google embeddings as features for a Logistic Regression classifier.
  - **Benefit**: Instead of a raw distance, you get a **Probability of Correctness**.
  - **Separability**: By training on the difference and product of embeddings, the model learns a much sharper decision boundary than simple cosine distance.
