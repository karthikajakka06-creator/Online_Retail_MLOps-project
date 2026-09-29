from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import FEATURE_COLUMNS, TARGET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPLIT_PATH = ARTIFACTS_DIR / "lab3_split.joblib"
BASELINE_MODEL_PATH = MODELS_DIR / "final_customer_purchase_model.pkl"
PRODUCTION_MODEL_PATH = MODELS_DIR / "production_customer_purchase_pipeline.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"


def ensure_project_directories() -> None:
    for directory in (ARTIFACTS_DIR, MODELS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_split(path: Path = SPLIT_PATH) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Training split not found: {path}; run preprocess.py first")
    return joblib.load(path)


def create_stratified_split(
    customer_features: pd.DataFrame,
) -> dict[str, object]:
    X = customer_features[FEATURE_COLUMNS]
    y = customer_features[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_customer_ids": customer_features.loc[X_test.index, "CustomerID"],
        "feature_columns": FEATURE_COLUMNS,
    }