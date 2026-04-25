import numpy as np

history = []

def check_drift(feature_vector):

    history.append(float(np.mean(feature_vector)))

    if len(history) < 50:
        return False

    if len(history) > 200:
        history.pop(0)

    old = np.mean(history[:25])
    new = np.mean(history[-25:])

    drift_detected = abs(old - new) > 0.5
    if drift_detected:
        print(f"[Monitor] Drift detected — old mean: {old:.3f}, new mean: {new:.3f}")

    return drift_detected

def reset_history():

    global history
    history = []
    print("[Monitor] Drift history reset after retrain.")