import pandas as pd
import numpy as np
import os
import time

os.makedirs("logs", exist_ok=True)

LOG_FILE = "logs/predictions.csv"


START_FILE = "logs/demo_start_time.txt"

if not os.path.exists(START_FILE):
    with open(START_FILE, "w") as f:
        f.write(str(time.time()))

with open(START_FILE, "r") as f:
    start_time = float(f.read().strip())

# grow at 3.3 rows/sec from when the server first started
seconds_running = time.time() - start_time
n = max(100, min(int(seconds_running * 3.3), 10000))  # cap at 10k for performance

np.random.seed(42)

predictions = np.zeros(n, dtype=int)

# inject fraud clusters proportional to data size
num_clusters = max(1, n // 500)
if n > 100:
    cluster_centers = np.random.choice(range(50, n - 50), num_clusters, replace=False)
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
print(f"Demo: {n} rows ({seconds_running:.0f}s running), {predictions.sum()} frauds")