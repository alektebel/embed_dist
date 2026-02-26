# Banking QA Validation Project

## Model Sizes
- **Base Model (multilingual-e5-large)**: ~2.2 GB
- **Metric Adapter**: ~3.1 MB
- **Dataset**: < 1 MB

## GitHub Upload Constraints
GitHub limits individual files to **100MB**. 
1. The **Code** and **Metric Adapter** are small and will be pushed to the repository.
2. The **Base Model (2.2GB)** exceeds the limit. It should be handled via **Git LFS** or downloaded manually.

## Offline Setup
1. Download the base model: `SentenceTransformer('intfloat/multilingual-e5-large').save('models/multilingual-e5-large')`
2. Ensure `models/metric_adapter.pt` is in the `models/` folder.
3. Run `validator.py`.
