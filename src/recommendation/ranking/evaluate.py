import numpy as np
import pandas as pd


def evaluate_ranker(
    dataframe: pd.DataFrame,
    score_column: str = "score",
    k: int = 20,
) -> dict:
    """
    End-to-end ranking evaluation.

    Important:
    Validation contains only genuinely retrieved candidates.

    Therefore:
    - if retrieval misses the target -> Recall@K = 0
    - if retrieval finds it but ranker puts it below K -> Recall@K = 0
    - only successful retrieve + rank contributes a hit
    """

    hits = []
    reciprocal_ranks = []

    total_sessions = 0
    sessions_with_target = 0

    for _, group in dataframe.groupby(
        "session",
        sort=False,
    ):
        total_sessions += 1

        # Does the candidate set contain the target at all?
        if group["label"].any():
            sessions_with_target += 1

        ranked = (
            group
            .sort_values(
                score_column,
                ascending=False,
            )
            .head(k)
        )

        labels = (
            ranked["label"]
            .to_numpy()
        )

        if labels.any():
            rank = (
                np.argmax(labels)
                + 1
            )

            hits.append(1.0)

            reciprocal_ranks.append(
                1.0 / rank
            )

        else:
            hits.append(0.0)
            reciprocal_ranks.append(
                0.0
            )

    candidate_recall = (
        sessions_with_target
        / total_sessions
        if total_sessions
        else 0.0
    )

    return {
        "candidate_recall":
            float(candidate_recall),

        f"end_to_end_recall_at_{k}":
            float(np.mean(hits)),

        f"end_to_end_mrr_at_{k}":
            float(
                np.mean(
                    reciprocal_ranks
                )
            ),
    }