import joblib
import numpy as np

MODEL_PATH = "models/model.pkl"


_model = joblib.load(MODEL_PATH)

def reload_model():

    global _model
    _model = joblib.load(MODEL_PATH)
    print("[Predict] Model reloaded from disk.")

def predict(data):
    arr = np.array(data).reshape(1, -1)
    return _model.predict(arr)[0]