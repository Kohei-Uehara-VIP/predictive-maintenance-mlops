"""Simulate production data with drift and detect it with an Evidently AI test suite."""
import json

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src import config
from src.data import load_raw_data, prepare_features, split_data


def simulate_production_data(X_holdout: pd.DataFrame) -> pd.DataFrame:
    """Hotter season: both temperatures rise by 4 K, and the torque sensor gets noisier."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    current = X_holdout.copy()
    current["air_temperature_k"] = current["air_temperature_k"] + 4.0
    current["process_temperature_k"] = current["process_temperature_k"] + 4.0
    current["torque_nm"] = current["torque_nm"] + rng.normal(0, 3, len(current))
    return current


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compare current data with the training reference and save an HTML report."""
    definition = DataDefinition(
        numerical_columns=config.NUMERIC_FEATURES,
        categorical_columns=config.CATEGORICAL_FEATURES,
    )
    report = Report([DataDriftPreset()], include_tests=True)
    snapshot = report.run(
        current_data=Dataset.from_pandas(current, data_definition=definition),
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
    )
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(config.DRIFT_REPORT_PATH))

    drifted_columns = []
    for test in snapshot.dict()["tests"]:
        status = getattr(test["status"], "value", test["status"])
        column = test["metric_config"]["params"].get("column")
        if column and status == "FAIL":
            drifted_columns.append(column)

    n_columns = len(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)
    return {
        "drift_detected": len(drifted_columns) > 0,
        "drifted_columns": drifted_columns,
        "drifted_share": round(len(drifted_columns) / n_columns, 3),
    }


def main() -> dict:
    X, y = prepare_features(load_raw_data())
    X_train, X_test, _, _ = split_data(X, y)

    current = simulate_production_data(X_test)
    current.to_csv(config.PRODUCTION_DATA_PATH, index=False)

    result = detect_drift(reference=X_train, current=current)
    config.DRIFT_SUMMARY_PATH.write_text(json.dumps(result, indent=2))
    print(f"Drift result: {result}")
    print(f"HTML report saved to {config.DRIFT_REPORT_PATH}")
    return result


if __name__ == "__main__":
    main()