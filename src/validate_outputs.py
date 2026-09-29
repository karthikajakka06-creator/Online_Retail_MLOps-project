import json

import joblib
import numpy as np
import pandas as pd

from data_utils import FEATURE_COLUMNS
from pipeline_utils import (
    ARTIFACTS_DIR,
    OUTPUTS_DIR,
    PREPROCESSOR_PATH,
    ensure_project_directories,
    load_split,
)


def main() -> None:
    print("Validating production preprocessing outputs...")
    ensure_project_directories()
    split = load_split()
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    expected = preprocessor.transform(split["X_test"])
    actual = pd.read_csv(OUTPUTS_DIR / "preprocessed_test.csv")
    if list(actual.columns) != FEATURE_COLUMNS:
        raise ValueError("Preprocessed output columns do not match the model feature contract")
    if actual.shape != (len(split["X_test"]), len(FEATURE_COLUMNS)):
        raise ValueError("Preprocessed output dimensions do not match the test split")
    if not np.isfinite(actual.to_numpy()).all():
        raise ValueError("Preprocessed output contains non-finite values")
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-6, atol=1e-6)
    report = {
        "valid": True,
        "train_shape": list(pd.read_csv(OUTPUTS_DIR / "preprocessed_train.csv").shape),
        "test_shape": list(actual.shape),
        "features": FEATURE_COLUMNS,
        "finite_values": True,
        "matches_fitted_preprocessor": True,
    }
    (ARTIFACTS_DIR / "preprocessing_output_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("PASS: dimensions, feature columns, finite values, and fitted transforms match.")


if __name__ == "__main__":
    main()