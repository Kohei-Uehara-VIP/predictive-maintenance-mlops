"""Prefect flow: check for data drift every day and retrain the model when drift is found."""
import argparse

from prefect import flow, get_run_logger, task
from prefect.schedules import Cron

from src import drift, export_model, train


@task(name="detect-data-drift")
def detect_drift_task() -> dict:
    return drift.main()


@task(name="retrain-model", retries=1)
def retrain_task() -> None:
    train.main()


@task(name="export-production-model")
def export_model_task() -> None:
    export_model.main()


@flow(name="drift-triggered-retraining", log_prints=True)
def retraining_flow(force_retrain: bool = False) -> dict:
    logger = get_run_logger()
    result = detect_drift_task()

    if result["drift_detected"] or force_retrain:
        logger.info(f"Drift detected in {result['drifted_columns']} -> retraining the model")
        retrain_task()
        export_model_task()
    else:
        logger.info("No drift detected -> keeping the current production model")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-once", action="store_true", help="run the flow now and exit")
    args = parser.parse_args()

    if args.run_once:
        retraining_flow()
    else:
        # Register a deployment that runs every day at 02:00 Japan time and keep serving it
        retraining_flow.serve(
            name="daily-drift-check",
            schedules=[Cron("0 2 * * *", timezone="Asia/Tokyo")],
        )