import pandas as pd
import numpy as np
import os

# generates a realistic pre-baked predictions.csv so the dashboard
# looks live even without the pipeline running on the server
np.random.seed(42)
os.makedirs("logs", exist_ok=True)

n = 2000

# simulate realistic fraud pattern — 0.17% base rate with occasional clusters
predictions = np.zeros(n, dtype=int)

# inject 4 fraud clusters at random positions (simulates drift periods)
for cluster_center in [300, 750, 1200, 1700]:
    for i in range(cluster_center - 3, cluster_center + 4):
        if 0 <= i < n:
            predictions[i] = 1

# base fraud rate — random sparse frauds
for i in range(n):
    if predictions[i] == 0 and np.random.random() < 0.001:
        predictions[i] = 1

# generate realistic feature values matching fraud.csv schema
V_cols = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
amounts = np.random.exponential(scale=88, size=n)

# spike amounts at fraud positions (makes the Amount chart look realistic)
for i in range(n):
    if predictions[i] == 1:
        amounts[i] = amounts[i] * np.random.uniform(5, 15)

df = pd.DataFrame(V_cols)
df["Amount"] = amounts
df["Class"] = 0
df["prediction"] = predictions

df.to_csv("logs/predictions.csv", index=False)
print(f"Demo data generated — {n} rows, {predictions.sum()} frauds ({predictions.mean()*100:.2f}%)")
