import json

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from pipeline_utils import ARTIFACTS_DIR, ensure_project_directories, load_split
from tracking_utils import configure_mlflow


def main() -> None:
    print("Starting tracked Random Forest training...")
    ensure_project_directories()
    configure_mlflow()
    split = load_split()
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(split["X_train"], split["y_train"])
    predictions = model.predict(split["X_test"])
    probabilities = model.predict_proba(split["X_test"])[:, 1]

    with mlflow.start_run(run_name="random-forest-baseline") as run:
        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "n_estimators": 100,
                "random_state": 42,
                "forecast_days": 30,
                "test_size": 0.2,
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": float(mlflow.models.evaluate.__module__ is not None)
                if False
                else 0.0
            }
        )
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        metrics = {
            "accuracy": float(accuracy_score(split["y_test"], predictions)),
            "precision": float(precision_score(split["y_test"], predictions, zero_division=0)),
            "recall": float(recall_score(split["y_test"], predictions, zero_division=0)),
            "f1": float(f1_score(split["y_test"], predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(split["y_test"], probabilities)),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=split["X_test"].head(5).astype("float64"),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        (ARTIFACTS_DIR / "mlflow_run.json").write_text(
            json.dumps({"run_id": run.info.run_id, "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
        print(f"Logged run {run.info.run_id}: F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()