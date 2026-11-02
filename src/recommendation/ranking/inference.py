import pandas as pd
import mlflow

from .bpr_retriever import BPRRetriever
from .config import (
    EVENTS_PATH,
    BPR_MODEL_PATH,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    CANDIDATES_PER_SESSION,
)
from .features import (
    FEATURE_COLUMNS,
    build_item_stats,
    build_candidate_features,
)


class RecommendationPipeline:
    """
    End-to-end recommendation inference pipeline.

    Flow:
        observed session
            ↓
        BPR retrieval
            ↓
        candidate feature engineering
            ↓
        LambdaMART champion ranker
            ↓
        final Top-K recommendations
    """

    def __init__(self):
        # -------------------------------------------------
        # 1. Configure MLflow
        # -------------------------------------------------
        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        # -------------------------------------------------
        # 2. Load BPR candidate generator
        # -------------------------------------------------
        print(
            f"Loading BPR retriever from "
            f"{BPR_MODEL_PATH}"
        )

        self.retriever = BPRRetriever(
            BPR_MODEL_PATH
        )

        # -------------------------------------------------
        # 3. Load the current champion ranker
        #
        # We deliberately DO NOT hard-code:
        #
        #     version=4
        #
        # The alias resolves whatever version is currently
        # promoted as champion.
        # -------------------------------------------------
        model_uri = (
            f"models:/"
            f"{REGISTERED_MODEL_NAME}"
            f"@champion"
        )

        print(
            f"Loading champion ranker: "
            f"{model_uri}"
        )

        self.ranker = (
            mlflow.lightgbm.load_model(
                model_uri
            )
        )

        # -------------------------------------------------
        # 4. Build item-level historical statistics
        #
        # For now we calculate them from the processed
        # OTTO event dataset once when the pipeline starts.
        #
        # Later:
        #     Feast / feature store
        # will replace this.
        # -------------------------------------------------
        print(
            f"Loading historical events from "
            f"{EVENTS_PATH}"
        )

        events = pd.read_parquet(
            EVENTS_PATH
        )

        print(
            "Building item statistics..."
        )

        self.item_stats = build_item_stats(
            events
        )

        print(
            "Recommendation pipeline ready."
        )

    def build_candidates(
        self,
        observed_aids,
        candidate_k=CANDIDATES_PER_SESSION,
    ):
        """
        Stage 1: candidate retrieval.

        Input:
            observed_aids = [45, 91, 102]

        Output example:
            [
                812,
                331,
                991,
                ...
            ]
        """

        recommendations = (
            self.retriever.recommend(
                observed_aids,
                k=candidate_k,
            )
        )

        return recommendations

    def build_feature_dataframe(
        self,
        observed_aids,
        candidates,
    ):
        """
        Build one LambdaMART feature row per candidate.
        """

        rows = []

        for rank, candidate_aid in enumerate(
            candidates,
            start=1,
        ):
            features = (
                build_candidate_features(
                    observed_aids=observed_aids,
                    candidate_aid=int(
                        candidate_aid
                    ),
                    bpr_rank=rank,
                    item_stats=self.item_stats,
                )
            )

            rows.append(
                {
                    "candidate_aid":
                        int(candidate_aid),

                    **features,
                }
            )

        return pd.DataFrame(
            rows
        )

    def recommend(
        self,
        observed_aids,
        top_k=20,
        candidate_k=CANDIDATES_PER_SESSION,
    ):
        """
        Full end-to-end recommendation.

        Example:

            pipeline.recommend(
                observed_aids=[45, 91, 102],
                top_k=10,
            )
        """

        # -------------------------------------------------
        # Stage 1:
        # BPR retrieves candidate items
        # -------------------------------------------------
        candidates = self.build_candidates(
            observed_aids=observed_aids,
            candidate_k=candidate_k,
        )

        if not candidates:
            return []

        # -------------------------------------------------
        # Stage 2:
        # Build LambdaMART features
        # -------------------------------------------------
        candidate_df = (
            self.build_feature_dataframe(
                observed_aids=
                    observed_aids,

                candidates=
                    candidates,
            )
        )

        # IMPORTANT:
        # Keep exactly the same feature order that was
        # used during training.
        X = candidate_df[
            FEATURE_COLUMNS
        ]

        # -------------------------------------------------
        # Stage 3:
        # LambdaMART produces ranking scores
        # -------------------------------------------------
        candidate_df[
            "ranking_score"
        ] = self.ranker.predict(
            X
        )

        # -------------------------------------------------
        # Stage 4:
        # Sort candidates by LambdaMART score
        # -------------------------------------------------
        ranked = (
            candidate_df
            .sort_values(
                "ranking_score",
                ascending=False,
            )
            .head(top_k)
        )

        # -------------------------------------------------
        # Return both item ID and ranker score.
        #
        # Score is NOT necessarily a probability.
        # It is primarily useful for ordering.
        # -------------------------------------------------
        results = []

        for position, row in enumerate(
            ranked.itertuples(
                index=False
            ),
            start=1,
        ):
            results.append(
                {
                    "rank": position,
                    "aid": int(
                        row.candidate_aid
                    ),
                    "score": float(
                        row.ranking_score
                    ),
                }
            )

        return results


def main():
    pipeline = RecommendationPipeline()

    # Temporary example.
    # Later this will come from a FastAPI request.
    observed_aids = [
        45,
        91,
        102,
    ]

    recommendations = (
        pipeline.recommend(
            observed_aids=
                observed_aids,

            top_k=10,
        )
    )

    print()
    print(
        f"Observed session: "
        f"{observed_aids}"
    )

    print()
    print(
        "Final recommendations:"
    )

    for recommendation in recommendations:
        print(
            f"#{recommendation['rank']:02d} "
            f"aid={recommendation['aid']} "
            f"score="
            f"{recommendation['score']:.4f}"
        )


if __name__ == "__main__":
    main()