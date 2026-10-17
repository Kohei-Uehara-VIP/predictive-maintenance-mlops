"""Check that the Docker API returns the same predictions as the local model."""
import joblib
import requests

from src import config
from src.data import load_raw_data, prepare_features, split_data

API_URL = "http://127.0.0.1:8000/predict"
N_SAMPLES = 200


def main() -> None:
    X, y = prepare_features(load_raw_data())
    _, X_test, _, _ = split_data(X, y)
    sample = X_test.head(N_SAMPLES)

    local = joblib.load(config.MODEL_PATH).predict_proba(sample)[:, 1].round(4)

    diffs = []
    for row, expected in zip(sample.to_dict(orient="records"), local):
        response = requests.post(API_URL, json=row, timeout=10)
        response.raise_for_status()
        diffs.append(abs(response.json()["failure_probability"] - expected))

    mismatches = sum(d > 1.0001e-4 for d in diffs)
    print(f"Compared {len(diffs)} rows | max abs diff = {max(diffs):.6f} | mismatches = {mismatches}")


if __name__ == "__main__":
    main()