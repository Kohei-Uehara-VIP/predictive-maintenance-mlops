"""Load the raw AI4I CSV, validate it, and split it into train/test sets."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and run basic sanity checks."""
    df = pd.read_csv(path)
    validate_raw_data(df)
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    """Fail fast if the file does not look like the AI4I dataset we expect."""
    missing_cols = set(config.EXPECTED_RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")
    if len(df) != config.EXPECTED_ROWS:
        raise ValueError(f"Expected {config.EXPECTED_ROWS} rows, got {len(df)}")


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop ID / leakage columns, rename columns, and separate X and y."""
    df = df.drop(columns=config.ID_COLUMNS + config.LEAKAGE_COLUMNS)
    df = df.rename(columns=config.RENAME_MAP)
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET]
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified split keeps the failure rate (~3.39%) the same in train and test."""
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )