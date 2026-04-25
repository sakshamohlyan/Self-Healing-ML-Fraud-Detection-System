import pandas as pd
import os
import threading

os.makedirs("logs", exist_ok=True)

LOG_FILE = "logs/predictions.csv"

# BUG FIX: Logger is called from the pipeline thread while retrain may also
# trigger activity. A lock ensures only one thread writes at a time,
# preventing corrupted/missing rows in the CSV.
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