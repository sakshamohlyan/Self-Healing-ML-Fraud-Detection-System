import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

MODEL_PATH = "models/model.pkl"

def retrain():
    # SUGGESTION 2a: Retrain on recent logged predictions (last 5000 rows)
    # so the model adapts to current data, not the original 284k row dataset.
    # Falls back to fraud.csv if not enough logged data yet.
    try:
        df = pd.read_csv("logs/predictions.csv").tail(5000)
        if len(df) < 100:
            raise ValueError("Not enough logged data yet")

        # logged CSV has both original columns + "prediction" column
        # drop "prediction" to get features, use it as the label
        X = df.drop(["Class", "prediction"], axis=1, errors="ignore")
        y = df["prediction"]
        print(f"[Retrain] Using {len(df)} rows from recent logs.")

    except Exception:
        df = pd.read_csv("data/fraud.csv")
        X = df.drop("Class", axis=1)
        y = df["Class"]
        print(f"[Retrain] Falling back to full fraud.csv ({len(df)} rows).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # SUGGESTION 2b: class_weight="balanced" was missing from retrain —
    # the retrained model was worse than the original. Now consistent with train.py.
    new_model = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42)
    new_model.fit(X_train, y_train)

    # SUGGESTION 5: Only save new model if it's at least as good as current one.
    # Prevents a bad retrain from silently degrading production performance.
    try:
        old_model = joblib.load(MODEL_PATH)
        score_before = f1_score(y_test, old_model.predict(X_test), zero_division=0)
        score_after  = f1_score(y_test, new_model.predict(X_test), zero_division=0)

        if score_after >= score_before:
            joblib.dump(new_model, MODEL_PATH)
            print(f"[Retrain] Model improved: F1 {score_before:.3f} → {score_after:.3f}. Saved.")
        else:
            print(f"[Retrain] New model worse (F1 {score_after:.3f} < {score_before:.3f}). Keeping old model.")

    except Exception:
        # No existing model to compare against — just save
        joblib.dump(new_model, MODEL_PATH)
        print("[Retrain] No previous model found. Saved new model.")