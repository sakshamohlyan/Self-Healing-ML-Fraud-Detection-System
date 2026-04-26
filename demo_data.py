import pandas as pd
import numpy as np
import os
import time

os.makedirs("logs", exist_ok=True)

LOG_FILE = "logs/predictions.csv"

# generate a row count that grows over time based on current timestamp
# this makes the dashboard look like a live pipeline is running
# 3.3 rows/sec starting from a fixed reference point
REFERENCE_TIME = 1700000000  # fixed past timestamp
rows_since_start = int((time.time() - REFERENCE_TIME) * 3.3)
n = min(rows_since_start, 50000)  # cap at 50k so page stays fast

np.random.seed(int(time.time()) // 300)  # changes seed every 5 mins

predictions = np.zeros(n, dtype=int)

# inject fraud clusters
num_clusters = max(1, n // 500)
cluster_centers = np.random.choice(range(50, n-50), num_clusters, replace=False)
for center in cluster_centers:
    for i in range(center - 3, center + 4):
        if 0 <= i < n:
            predictions[i] = 1

# sparse random frauds
for i in range(n):
    if predictions[i] == 0 and np.random.random() < 0.001:
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
print(f"Demo data: {n} rows, {predictions.sum()} frauds")