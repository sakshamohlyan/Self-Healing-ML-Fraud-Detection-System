import pandas as pd
import numpy as np
import os

os.makedirs("logs", exist_ok=True)

LOG_FILE = "logs/predictions.csv"

def generate(n):
    """Generate n rows of realistic fraud detection data."""
    np.random.seed(42)

    predictions = np.zeros(n, dtype=int)

    # inject fraud clusters
    num_clusters = max(1, n // 500)
    if n > 100:
        centers = np.random.choice(range(50, n - 50), num_clusters, replace=False)
        for center in centers:
            for i in range(center - 3, center + 4):
                if 0 <= i < n:
                    predictions[i] = 1

    # sparse random frauds — ~0.1% base rate
    rng = np.random.default_rng(n)  # seed by n so same n = same data
    for i in range(n):
        if predictions[i] == 0 and rng.random() < 0.001:
            predictions[i] = 1

    V_cols = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    amounts = np.random.exponential(scale=88, size=n)

    for i in range(n):
        if predictions[i] == 1:
            amounts[i] = amounts[i] * np.random.uniform(5, 15)

    df = pd.DataFrame(V_cols)
    df["Amount"] = amounts
    df["Class"] = 0
    df["prediction"] = predictions

    df.to_csv(LOG_FILE, index=False)
    return df