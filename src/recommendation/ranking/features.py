import pandas as pd

FEATURE_COLUMNS = [
    "session_length",
    "candidate_seen_before",
    "item_interactions",
    "item_clicks",
    "item_carts",
    "item_orders",
    "item_cart_rate",
    "item_order_rate",
    "bpr_rank",
    "bpr_rr",
]


def build_item_stats(events: pd.DataFrame) -> pd.DataFrame:
    item_stats = (
        events.groupby("aid")
        .agg(
            item_interactions=("aid", "size"),
            item_last_ts=("ts", "max"),
        )
    )

    event_counts = (
        events.groupby(["aid", "type"])
        .size()
        .unstack(fill_value=0)
    )

    item_stats = item_stats.join(event_counts, how="left").fillna(0)

    for col in ["clicks", "carts", "orders"]:
        if col not in item_stats.columns:
            item_stats[col] = 0

    item_stats["cart_rate"] = (
        item_stats["carts"] / item_stats["clicks"].clip(lower=1)
    )
    item_stats["order_rate"] = (
        item_stats["orders"] / item_stats["clicks"].clip(lower=1)
    )

    return item_stats


def build_candidate_features(
    observed_aids,
    candidate_aid: int,
    bpr_rank: int | None,
    item_stats: pd.DataFrame,
) -> dict:
    observed_set = set(observed_aids)

    features = {
        "session_length": len(observed_aids),
        "candidate_seen_before": int(candidate_aid in observed_set),
        "item_interactions": 0.0,
        "item_clicks": 0.0,
        "item_carts": 0.0,
        "item_orders": 0.0,
        "item_cart_rate": 0.0,
        "item_order_rate": 0.0,
        "bpr_rank": 999.0,
        "bpr_rr": 0.0,
    }

    if candidate_aid in item_stats.index:
        stats = item_stats.loc[candidate_aid]
        features.update({
            "item_interactions": float(stats["item_interactions"]),
            "item_clicks": float(stats["clicks"]),
            "item_carts": float(stats["carts"]),
            "item_orders": float(stats["orders"]),
            "item_cart_rate": float(stats["cart_rate"]),
            "item_order_rate": float(stats["order_rate"]),
        })

    if bpr_rank is not None:
        features["bpr_rank"] = float(bpr_rank)
        features["bpr_rr"] = 1.0 / float(bpr_rank)

    return features
