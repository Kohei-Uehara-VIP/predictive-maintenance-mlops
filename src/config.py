"""Central configuration: paths, column names and constants."""
from pathlib import Path

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "ai4i2020.csv"

# ---- Reproducibility ----
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---- Column handling ----
# Original CSV column names -> clean snake_case names (easier for code, API and XGBoost)
RENAME_MAP = {
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}

# Columns removed before training
ID_COLUMNS = ["UDI", "Product ID"]                     # unique per row -> no predictive value
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # failure-mode results -> data leakage

TARGET = "machine_failure"
NUMERIC_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
CATEGORICAL_FEATURES = ["type"]

# Sanity check: every raw file must contain these columns and rows
EXPECTED_RAW_COLUMNS = ID_COLUMNS + list(RENAME_MAP.keys()) + LEAKAGE_COLUMNS
EXPECTED_ROWS = 10_000