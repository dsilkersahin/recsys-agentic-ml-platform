import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient
import pandas as pd

from .config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    RANK_TRAIN_PATH,
    RANK_VALID_PATH,
    MODEL_DIR,
    RANDOM_SEED,
)

REGISTERED_MODEL_NAME = "otto-ranker"

from .features import FEATURE_COLUMNS
from .model import build_ranker
from .evaluate import evaluate_ranker


def configure_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def load_data():
    return (
        pd.read_parquet(RANK_TRAIN_PATH),
        pd.read_parquet(RANK_VALID_PATH),
    )


def train_ranker(train_df, valid_df):
    train_df = train_df.sort_values("session")
    valid_df = valid_df.sort_values("session")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]

    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df["label"]

    train_groups = (
        train_df.groupby("session", sort=False).size().to_numpy()
    )

    valid_groups = (
        valid_df.groupby("session", sort=False).size().to_numpy()
    )

    params = {
    "n_estimators": 600,
    "learning_rate": 0.003,    # changed
    "num_leaves": 31,         # changed
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    }

    ranker = build_ranker(**params)

    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")

        mlflow.log_params(params)

        mlflow.log_params({
            "feature_count": len(FEATURE_COLUMNS),
            "train_rows": len(train_df),
            "valid_rows": len(valid_df),
            "train_sessions": train_df["session"].nunique(),
            "valid_sessions": valid_df["session"].nunique(),
        })

        mlflow.set_tags({
            "model_type": "LambdaMART",
            "task": "recommendation_ranking",
            "candidate_generator": "BPR",
            "dataset": "OTTO",
        })

        ranker.fit(
            X_train,
            y_train,
            group=train_groups,
            eval_set=[(X_valid, y_valid)],
            eval_group=[valid_groups],
            eval_at=[5, 10, 20],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(50),
            ],
        )

        scored_valid = valid_df.copy()
        scored_valid["score"] = ranker.predict(X_valid)

        metrics = evaluate_ranker(scored_valid, k=20)
        mlflow.log_metrics(metrics)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        local_model_path = MODEL_DIR / "lambdamart_ranker.txt"
        ranker.booster_.save_model(str(local_model_path))

        mlflow.log_artifact(
            str(local_model_path),
            artifact_path="native_model",
        )

        model_info = mlflow.lightgbm.log_model(
            ranker,
            name="ranker",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="skops",
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMRanker",
            ],
        )

        print(f"Metrics: {metrics}")

    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    latest_version = max(
        versions,
        key=lambda v: int(v.version),
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=latest_version.version,
    )

    print(
        f"Champion -> "
        f"{REGISTERED_MODEL_NAME} "
        f"v{latest_version.version}"
    )

    return ranker, metrics


def main():
    configure_mlflow()

    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")

    train_df, valid_df = load_data()

    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(valid_df):,}")

    train_ranker(train_df, valid_df)


if __name__ == "__main__":
    main()
