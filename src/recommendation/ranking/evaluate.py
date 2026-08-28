import numpy as np
import pandas as pd


def evaluate_ranker(
    dataframe: pd.DataFrame,
    score_column: str = "score",
    k: int = 20,
) -> dict:
    hits = []
    reciprocal_ranks = []

    for _, group in dataframe.groupby("session", sort=False):
        ranked = (
            group
            .sort_values(score_column, ascending=False)
            .head(k)
        )

        labels = ranked["label"].to_numpy()

        if labels.any():
            rank = np.argmax(labels) + 1
            hits.append(1.0)
            reciprocal_ranks.append(1.0 / rank)
        else:
            hits.append(0.0)
            reciprocal_ranks.append(0.0)

    return {
        f"recall_at_{k}": float(np.mean(hits)),
        f"mrr_at_{k}": float(np.mean(reciprocal_ranks)),
    }
