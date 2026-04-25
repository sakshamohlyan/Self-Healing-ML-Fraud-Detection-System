import joblib
import numpy as np

MODEL_PATH = "models/model.pkl"

# Cache model in memory — only reloaded when retrain explicitly calls reload_model()
# This avoids hitting the disk 3.3 times per second
_model = joblib.load(MODEL_PATH)

def reload_model():
    """Called by app.py after retrain finishes to pull new model into memory."""
    global _model
    _model = joblib.load(MODEL_PATH)
    print("[Predict] Model reloaded from disk.")

def predict(data):
    arr = np.array(data).reshape(1, -1)
    return _model.predict(arr)[0]