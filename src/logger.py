import pandas as pd
import os
import threading

os.makedirs("logs", exist_ok=True)

LOG_FILE = "logs/predictions.csv"

_lock = threading.Lock()

def log(data, prediction):
    row = data.to_dict()
    row["prediction"] = int(prediction)

    df = pd.DataFrame([row])

    with _lock:
        if not os.path.exists(LOG_FILE):
            df.to_csv(LOG_FILE, index=False)
        else:
            df.to_csv(LOG_FILE, mode='a', header=False, index=False)