import pandas as pd
import time
import queue
import random


data_queue = queue.Queue(maxsize=100)

def start_stream():
    df = pd.read_csv("data/fraud.csv")

    while True:

        for row in df.itertuples(index=False):
            row = pd.Series(row._asdict())

            if random.random() < 0.05:   # 5% chance of injected anomaly
                row = row.copy()
                row["Amount"] = row["Amount"] * random.uniform(5, 15)
                row["V1"] += random.uniform(-10, 10)
                row["V2"] += random.uniform(-10, 10)
                row["V3"] += random.uniform(-10, 10)

            data_queue.put(row)
            time.sleep(0.3)

def get_data():
    if not data_queue.empty():
        return data_queue.get()
    return None