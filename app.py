from fastapi import FastAPI
import threading

from src.stream import start_stream, get_data
from src.predict import predict, reload_model
from src.monitor import check_drift, reset_history
from src.retrain import retrain
from src.logger import log

app = FastAPI()

running = True
_retraining = False

def run_retrain():
    """Retrains in a background thread so pipeline never blocks."""
    global _retraining
    print("[Retrain] Starting background retraining...")
    retrain()
    reload_model()    # SUGGESTION 1: pull new model into memory after retrain
    reset_history()   # reset drift window so it doesn't re-trigger immediately
    _retraining = False
    print("[Retrain] Done. Pipeline continues normally.")

def pipeline():
    global _retraining

    while running:
        data = get_data()

        if data is None:
            continue

        # data is a pandas Series from itertuples → drop "Class" label only
        features = data.drop("Class").values
        prediction = predict(features)

        log(data, prediction)

        drift = check_drift(features)

        print("Prediction:", prediction, "| Drift:", drift)

        if drift and not _retraining:
            print("[Pipeline] Drift detected — launching background retrain")
            _retraining = True
            threading.Thread(target=run_retrain, daemon=True).start()

threading.Thread(target=start_stream, daemon=True).start()
threading.Thread(target=pipeline, daemon=True).start()

@app.get("/")
def home():
    return {"status": "running"}