import mlflow
from mlflow import MlflowClient

from .config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)


MIN_CANDIDATE_RECALL = 0.20
MIN_END_TO_END_RECALL_AT_20 = 0.10
MIN_END_TO_END_MRR_AT_20 = 0.03


from mlflow.exceptions import MlflowException


def get_champion_version(client):
    try:
        return client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
        )
    except MlflowException:
        return None

def get_candidate_version(client):
    try:
        return client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME,
            alias="candidate",
        )

    except MlflowException as exc:
        raise RuntimeError(
            f"No 'candidate' alias exists for "
            f"{REGISTERED_MODEL_NAME}. "
            f"Assign a model version to the candidate alias first."
        ) from exc


def is_better_than_champion(
    candidate_metrics,
    champion_metrics,
):
    """
    Simple first comparison rule.

    Candidate must not reduce retrieval recall,
    and should improve ranking quality.
    """

    candidate_recall = candidate_metrics.get(
        "candidate_recall",
        0.0,
    )

    candidate_e2e_recall = candidate_metrics.get(
        "end_to_end_recall_at_20",
        0.0,
    )

    candidate_mrr = candidate_metrics.get(
        "end_to_end_mrr_at_20",
        0.0,
    )

    champion_recall = champion_metrics.get(
        "candidate_recall",
        0.0,
    )

    champion_e2e_recall = champion_metrics.get(
        "end_to_end_recall_at_20",
        0.0,
    )

    champion_mrr = champion_metrics.get(
        "end_to_end_mrr_at_20",
        0.0,
    )

    print("\nCandidate vs Champion")

    print(
        f"candidate_recall: "
        f"{candidate_recall:.4f} vs "
        f"{champion_recall:.4f}"
    )

    print(
        f"end_to_end_recall_at_20: "
        f"{candidate_e2e_recall:.4f} vs "
        f"{champion_e2e_recall:.4f}"
    )

    print(
        f"end_to_end_mrr_at_20: "
        f"{candidate_mrr:.4f} vs "
        f"{champion_mrr:.4f}"
    )

    return (
        candidate_e2e_recall
        >= champion_e2e_recall
        and candidate_mrr
        > champion_mrr
    )

def get_run_metrics(client, model_version):
    """
    A registered model version stores the run_id
    that produced it.

    We use that run_id to get validation metrics.
    """

    run = client.get_run(
        model_version.run_id
    )

    return run.data.metrics


def passes_thresholds(metrics):
    """
    First simple validation gate.

    Later we can make this much more sophisticated.
    """

    candidate_recall = metrics.get(
        "candidate_recall",
        0.0,
    )

    recall20 = metrics.get(
        "end_to_end_recall_at_20",
        0.0,
    )

    mrr20 = metrics.get(
        "end_to_end_mrr_at_20",
        0.0,
    )

    print(
        f"candidate_recall: {candidate_recall:.4f}"
    )

    print(
        f"end_to_end_recall_at_20: {recall20:.4f}"
    )

    print(
        f"end_to_end_mrr_at_20: {mrr20:.4f}"
    )

    return (
        candidate_recall
        >= MIN_CANDIDATE_RECALL

        and recall20
        >= MIN_END_TO_END_RECALL_AT_20

        and mrr20
        >= MIN_END_TO_END_MRR_AT_20
    )


def promote(client, version):
    """
    Move champion alias to this model version.
    """

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=version,
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="validation_status",
        value="passed",
    )


def reject(client, version):
    """
    Keep the model registered, but mark it as failed.
    """

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="validation_status",
        value="failed",
    )

def main():
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    candidate = get_candidate_version(
        client
    )

    candidate_metrics = get_run_metrics(
        client,
        candidate,
    )

    print(
        f"Evaluating candidate "
        f"v{candidate.version}"
    )

    # Gate 1:
    # minimum acceptable quality
    if not passes_thresholds(
        candidate_metrics
    ):
        reject(
            client,
            candidate.version,
        )

        print("Candidate failed thresholds.")
        return

    # Gate 2:
    # compare against current champion
    champion = get_champion_version(
        client
    )

    # No champion yet:
    # candidate becomes first champion.
    if champion is None:
        promote(
            client,
            candidate.version,
        )

        print(
            f"First champion: "
            f"v{candidate.version}"
        )

        return

    champion_metrics = get_run_metrics(
        client,
        champion,
    )

    if is_better_than_champion(
        candidate_metrics,
        champion_metrics,
    ):
        promote(
            client,
            candidate.version,
        )

        print(
            f"PROMOTED: "
            f"v{candidate.version} "
            f"replaces v{champion.version}"
        )

    else:
        reject(
            client,
            candidate.version,
        )

        print(
            f"REJECTED: "
            f"v{candidate.version}. "
            f"Champion remains "
            f"v{champion.version}."
        )

if __name__ == "__main__":
    print(
        f"MLflow URI: "
        f"{MLFLOW_TRACKING_URI}"
    )
    main()
    print("Done.")