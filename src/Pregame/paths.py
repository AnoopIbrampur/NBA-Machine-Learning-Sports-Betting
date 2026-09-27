from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
DATASET_TABLE = "dataset_2012-26"
SPLIT_KEYS = BASE_DIR / "Data" / "splits" / "split_keys.csv"

# Regenerable artifacts (gitignored).
WORK_DIR = BASE_DIR / "Data" / "pregame"
GAME_LOGS_DB = WORK_DIR / "GameLogs.sqlite"
MODEL_DIR = WORK_DIR / "models"

# Small outputs kept in git for the report.
RESULTS_DIR = BASE_DIR / "results" / "pregame"

GAME_KEY = ["Date", "TEAM_NAME", "TEAM_NAME.1"]  # game date, home team, away team


def features_path(window):
    return WORK_DIR / f"pregame_features_w{window}.csv"


def model_path(feature_set, window):
    return MODEL_DIR / f"xgb_{feature_set}_w{window}.json"
