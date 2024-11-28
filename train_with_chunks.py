import pandas as pd
import os
import subprocess
from sklearn.cluster import KMeans
import joblib
import json

# Parameters
DATA_FILE = "src/data/recipe_full.csv"
CHUNK_SIZE = 200000
MODEL_DIR = "models"
METRICS_FILE = "metrics.json"

def train_cluster_model(chunk, chunk_index):
    # Prepare data (remove target column if present in clustering)
    data = chunk.drop("target", axis=1, errors="ignore")

    # Train model
    model = KMeans(n_clusters=5, random_state=42)
    model.fit(data)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"model_chunk_{chunk_index}.joblib")
    joblib.dump(model, model_path)

    # Save metrics (e.g., inertia for clustering)
    metrics = {"chunk_index": chunk_index, "inertia": model.inertia_}
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f)

    return model_path, METRICS_FILE

def main():
    # Read data in chunks
    chunk_index = 0
    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
        print(f"Processing chunk {chunk_index}...")

        # Train clustering model
        model_path, metrics_path = train_cluster_model(chunk, chunk_index)

        # Track with DVC
        subprocess.run(["dvc", "add", model_path])
        subprocess.run(["git", "add", model_path + ".dvc", METRICS_FILE])
        subprocess.run(["git", "commit", "-m", f'Track model and metrics for chunk {chunk_index}'])

        chunk_index += 1

if __name__ == "__main__":
    main()
