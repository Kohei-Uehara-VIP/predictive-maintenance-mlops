"""Train Random Forest and XGBoost, log them to MLflow, and register the best model."""
import matplotlib

matplotlib.use("Agg")  # save plots to files without opening a window
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.data import load_raw_data, prepare_features, split_data
from src.preprocessing import build_preprocessor

THRESHOLD = 0.5  # probability above this -> predict "failure"


def get_models(y_train: pd.Series) -> dict:
    """Return the candidate models and the parameters to log for each one."""
    # Imbalance ratio: about 28 non-failures for every failure
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    rf_params = {"n_estimators": 300, "min_samples_leaf": 2, "class_weight": "balanced"}
    xgb_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1,
        "scale_pos_weight": round(scale_pos_weight, 2),
    }
    return {
        "random_forest": (
            RandomForestClassifier(**rf_params, random_state=config.RANDOM_STATE, n_jobs=-1),
            rf_params,
        ),
        "xgboost": (
            XGBClassifier(
                **xgb_params,
                eval_metric="logloss",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
            xgb_params,
        ),
    }


def evaluate(y_true: pd.Series, proba) -> dict:
    """Compute metrics that are meaningful for an imbalanced binary problem."""
    pred = (proba >= THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def log_plots(pipeline: Pipeline, y_test: pd.Series, proba) -> None:
    """Log ROC curve, confusion matrix and feature importance plots to MLflow."""
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
    ax.set_title("ROC curve")
    mlflow.log_figure(fig, "plots/roc_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, (proba >= THRESHOLD).astype(int), ax=ax)
    ax.set_title("Confusion matrix (threshold = 0.5)")
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)

    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    series = pd.Series(importances, index=names).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    series.plot.barh(ax=ax)
    ax.set_title("Feature importance")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/feature_importance.png")
    plt.close(fig)


def main() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    X, y = prepare_features(load_raw_data())
    X_train, X_test, y_train, y_test = split_data(X, y)

    results = []
    for name, (model, params) in get_models(y_train).items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, proba)

            mlflow.log_params({"model_type": name, **params})
            mlflow.log_metrics(metrics)
            log_plots(pipeline, y_test, proba)

            signature = infer_signature(X_test, pipeline.predict(X_test))
            model_info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=X_test.head(3),
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            results.append((name, metrics[config.PRIMARY_METRIC], model_info.model_uri))
            print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    best_name, best_score, best_uri = max(results, key=lambda r: r[1])
    version = mlflow.register_model(best_uri, config.REGISTERED_MODEL_NAME)
    MlflowClient().set_registered_model_alias(
        config.REGISTERED_MODEL_NAME, config.MODEL_ALIAS, version.version
    )
    print(
        f"Best model: {best_name} ({config.PRIMARY_METRIC}={best_score:.4f}) -> "
        f"registered as '{config.REGISTERED_MODEL_NAME}' v{version.version} "
        f"with alias '{config.MODEL_ALIAS}'"
    )


if __name__ == "__main__":
    main()