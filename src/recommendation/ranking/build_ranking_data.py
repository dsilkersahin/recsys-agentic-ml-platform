import numpy as np
import pandas as pd

from .bpr_retriever import BPRRetriever
from .config import (
    EVENTS_PATH,
    VALID_EXAMPLES_PATH,
    BPR_MODEL_PATH,
    RANK_TRAIN_PATH,
    RANK_VALID_PATH,
    MAX_RANKING_SESSIONS,
    CANDIDATES_PER_SESSION,
    RANDOM_SEED,
)
from .features import build_item_stats, build_candidate_features


def build_ranking_dataframe(events, valid_examples, retriever):
    item_stats = build_item_stats(events)
    rows = []

    subset = valid_examples.head(MAX_RANKING_SESSIONS)

    for session_idx, row in enumerate(
        subset.itertuples(index=False),
        start=1,
    ):
        recommendations = retriever.recommend(
            row.observed_aids,
            k=CANDIDATES_PER_SESSION,
        )

        candidate_map = {
            int(aid): rank
            for rank, aid in enumerate(recommendations, start=1)
        }

        candidate_map.setdefault(int(row.target_aid), None)

        for candidate_aid, bpr_rank in candidate_map.items():
            features = build_candidate_features(
                observed_aids=row.observed_aids,
                candidate_aid=candidate_aid,
                bpr_rank=bpr_rank,
                item_stats=item_stats,
            )

            rows.append({
                "session": row.session,
                "candidate_aid": candidate_aid,
                "label": int(candidate_aid == row.target_aid),
                **features,
            })

        if session_idx % 500 == 0:
            print(f"Built ranking candidates for {session_idx:,} sessions")

    return pd.DataFrame(rows)


def split_ranking_dataframe(ranking_df):
    sessions = ranking_df["session"].drop_duplicates().to_numpy()

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(sessions)

    split_idx = int(len(sessions) * 0.8)

    train_sessions = set(sessions[:split_idx])
    valid_sessions = set(sessions[split_idx:])

    rank_train = ranking_df[
        ranking_df["session"].isin(train_sessions)
    ].copy()

    rank_valid = ranking_df[
        ranking_df["session"].isin(valid_sessions)
    ].copy()

    return rank_train, rank_valid


def main():
    print(f"Loading events: {EVENTS_PATH}")
    events = pd.read_parquet(EVENTS_PATH)

    print(f"Loading validation examples: {VALID_EXAMPLES_PATH}")
    valid_examples = pd.read_parquet(VALID_EXAMPLES_PATH)

    print(f"Loading BPR model: {BPR_MODEL_PATH}")
    retriever = BPRRetriever(BPR_MODEL_PATH)

    ranking_df = build_ranking_dataframe(
        events,
        valid_examples,
        retriever,
    )

    rank_train, rank_valid = split_ranking_dataframe(ranking_df)

    RANK_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    rank_train.to_parquet(RANK_TRAIN_PATH, index=False)
    rank_valid.to_parquet(RANK_VALID_PATH, index=False)

    print(f"Saved train: {RANK_TRAIN_PATH} {rank_train.shape}")
    print(f"Saved valid: {RANK_VALID_PATH} {rank_valid.shape}")


if __name__ == "__main__":
    main()
