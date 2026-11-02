from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REGISTERED_MODEL_NAME = "otto-ranker"

EVENTS_PATH = PROCESSED_DIR / "otto_train_300k_sessions.parquet"
VALID_EXAMPLES_PATH = PROCESSED_DIR / "otto_validation_300k_sessions.parquet"
BPR_MODEL_PATH = MODEL_DIR / "bpr_model.pkl"

RANK_TRAIN_PATH = PROCESSED_DIR / "ranking_train.parquet"
RANK_VALID_PATH = PROCESSED_DIR / "ranking_valid.parquet"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "otto-recommendation-ranking",
)

MAX_RANKING_SESSIONS = int(os.getenv("MAX_RANKING_SESSIONS", "5000"))
CANDIDATES_PER_SESSION = int(os.getenv("CANDIDATES_PER_SESSION", "50"))
RANDOM_SEED = 42
