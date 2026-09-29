import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipeline_utils import BASELINE_MODEL_PATH, OUTPUTS_DIR, ensure_project_directories, load_split


def main() -> None:
    print("Starting baseline model evaluation...")
    ensure_project_directories()
    split = load_split()
    if not BASELINE_MODEL_PATH.is_file():
        raise FileNotFoundError("Baseline model is missing; run train.py first")
    model = joblib.load(BASELINE_MODEL_PATH)
    predictions = model.predict(split["X_test"])
    probabilities = model.predict_proba(split["X_test"])[:, 1]
    actual = split["y_test"]
    metrics = {
        "accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "f1": float(f1_score(actual, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
    }
    report = {
        **metrics,
        "classification_report": classification_report(
            actual, predictions, output_dict=True, zero_division=0
        ),
        "test_rows": int(len(actual)),
    }
    (OUTPUTS_DIR / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        confusion_matrix(actual, predictions),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(OUTPUTS_DIR / "confusion_matrix.csv")
    pd.DataFrame(
        {
            "CustomerID": split["test_customer_ids"].to_numpy(),
            "actual": actual.to_numpy(),
            "predicted": predictions,
            "purchase_probability": probabilities,
        }
    ).query("actual != predicted").to_csv(
        OUTPUTS_DIR / "error_analysis.csv", index=False
    )
    print("\n--- Evaluation Report ---")
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title():10}: {value:.4f}")
    print("Evaluation artifacts saved under outputs/.")


if __name__ == "__main__":
    main()