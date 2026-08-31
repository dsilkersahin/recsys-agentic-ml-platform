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
from .features import (
    build_item_stats,
    build_candidate_features,
)


def split_sessions(valid_examples: pd.DataFrame):
    """
    Split ranking queries BEFORE candidate generation.

    This is important because:
    - ranking TRAIN may use oracle target insertion
    - ranking VALIDATION must contain only genuinely retrieved candidates
    """

    examples = (
        valid_examples
        .head(MAX_RANKING_SESSIONS)
        .copy()
    )

    session_ids = (
        examples["session"]
        .drop_duplicates()
        .to_numpy()
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rng.shuffle(session_ids)

    split_idx = int(
        len(session_ids) * 0.8
    )

    train_session_ids = set(
        session_ids[:split_idx]
    )

    valid_session_ids = set(
        session_ids[split_idx:]
    )

    train_examples = examples[
        examples["session"].isin(
            train_session_ids
        )
    ].copy()

    valid_examples = examples[
        examples["session"].isin(
            valid_session_ids
        )
    ].copy()

    return (
        train_examples,
        valid_examples,
    )


def build_ranking_dataframe(
    examples: pd.DataFrame,
    retriever: BPRRetriever,
    item_stats: pd.DataFrame,
    ensure_positive: bool,
):
    """
    Create candidate rows for LambdaMART.

    ensure_positive=True:
        Used for ranker TRAINING.
        If retrieval misses the target, insert it so that
        the ranking query still contains one positive example.

    ensure_positive=False:
        Used for VALIDATION.
        Never inject the true target.
        This gives us honest end-to-end evaluation.
    """

    rows = []

    retrieval_hits = 0

    total_sessions = 0

    for session_idx, row in enumerate(
        examples.itertuples(index=False),
        start=1,
    ):
        total_sessions += 1

        recommendations = (
            retriever.recommend(
                row.observed_aids,
                k=CANDIDATES_PER_SESSION,
            )
        )

        # Candidate -> retrieval rank
        candidate_map = {
            int(aid): rank
            for rank, aid in enumerate(
                recommendations,
                start=1,
            )
        }

        target_aid = int(
            row.target_aid
        )

        # Measure genuine BPR retrieval performance
        target_retrieved = (
            target_aid
            in candidate_map
        )

        if target_retrieved:
            retrieval_hits += 1

        # TRAIN ONLY:
        # LambdaMART needs positive examples.
        #
        # Crucially, we do NOT give the injected target
        # a fake rank such as 999 as a meaningful feature.
        #
        # It receives the normal "not retrieved" feature values.
        if (
            ensure_positive
            and not target_retrieved
        ):
            candidate_map[
                target_aid
            ] = None

        for (
            candidate_aid,
            bpr_rank,
        ) in candidate_map.items():

            features = (
                build_candidate_features(
                    observed_aids=
                        row.observed_aids,

                    candidate_aid=
                        candidate_aid,

                    bpr_rank=
                        bpr_rank,

                    item_stats=
                        item_stats,
                )
            )

            rows.append(
                {
                    "session":
                        row.session,

                    "candidate_aid":
                        candidate_aid,

                    "label":
                        int(
                            candidate_aid
                            == target_aid
                        ),

                    **features,
                }
            )

        if (
            session_idx
            % 500
            == 0
        ):
            print(
                f"Processed "
                f"{session_idx:,} sessions"
            )

    ranking_df = pd.DataFrame(
        rows
    )

    candidate_recall = (
        retrieval_hits
        / total_sessions
        if total_sessions
        else 0.0
    )

    return (
        ranking_df,
        candidate_recall,
    )


def main():
    print(
        f"Loading events: "
        f"{EVENTS_PATH}"
    )

    events = pd.read_parquet(
        EVENTS_PATH
    )

    print(
        f"Loading validation examples: "
        f"{VALID_EXAMPLES_PATH}"
    )

    examples = pd.read_parquet(
        VALID_EXAMPLES_PATH
    )

    print(
        f"Loading BPR: "
        f"{BPR_MODEL_PATH}"
    )

    retriever = BPRRetriever(
        BPR_MODEL_PATH
    )

    print(
        "Building item statistics..."
    )

    item_stats = build_item_stats(
        events
    )

    (
        rank_train_examples,
        rank_valid_examples,
    ) = split_sessions(
        examples
    )

    print()
    print(
        "Building RANKING TRAIN..."
    )

    (
        rank_train,
        train_candidate_recall,
    ) = build_ranking_dataframe(
        examples=
            rank_train_examples,

        retriever=
            retriever,

        item_stats=
            item_stats,

        ensure_positive=True,
    )

    print()
    print(
        "Building RANKING VALIDATION..."
    )

    (
        rank_valid,
        valid_candidate_recall,
    ) = build_ranking_dataframe(
        examples=
            rank_valid_examples,

        retriever=
            retriever,

        item_stats=
            item_stats,

        ensure_positive=False,
    )

    RANK_TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rank_train.to_parquet(
        RANK_TRAIN_PATH,
        index=False,
    )

    rank_valid.to_parquet(
        RANK_VALID_PATH,
        index=False,
    )

    print()
    print(
        f"Train shape: "
        f"{rank_train.shape}"
    )

    print(
        f"Validation shape: "
        f"{rank_valid.shape}"
    )

    print()
    print(
        f"Train candidate Recall@"
        f"{CANDIDATES_PER_SESSION}: "
        f"{train_candidate_recall:.4f}"
    )

    print(
        f"Validation candidate Recall@"
        f"{CANDIDATES_PER_SESSION}: "
        f"{valid_candidate_recall:.4f}"
    )

    print()
    print(
        f"Saved: "
        f"{RANK_TRAIN_PATH}"
    )

    print(
        f"Saved: "
        f"{RANK_VALID_PATH}"
    )


if __name__ == "__main__":
    main()