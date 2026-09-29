import json

from data_utils import FEATURES_PATH, TARGET, build_customer_features, load_transactions
from pipeline_utils import ARTIFACTS_DIR, SPLIT_PATH, create_stratified_split, ensure_project_directories
import joblib


def main() -> None:
    print("Starting Online Retail preprocessing...")
    ensure_project_directories()
    transactions = load_transactions()
    customer_features, cutoff_date = build_customer_features(transactions)
    split = create_stratified_split(customer_features)

    customer_features.to_csv(FEATURES_PATH, index=False)
    joblib.dump(split, SPLIT_PATH)
    summary = {
        "source": "data/OnlineRetail.csv",
        "cleaned_transactions": int(len(transactions)),
        "customer_count": int(len(customer_features)),
        "forecast_days": 30,
        "cutoff_date": cutoff_date.isoformat(),
        "features": split["feature_columns"],
        "target": TARGET,
        "target_counts": {
            str(label): int(count)
            for label, count in customer_features[TARGET].value_counts().sort_index().items()
        },
        "train_rows": int(len(split["y_train"])),
        "test_rows": int(len(split["y_test"])),
        "random_state": 42,
    }
    (ARTIFACTS_DIR / "preprocessing_summary_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Created {len(customer_features):,} customer rows from {len(transactions):,} transactions.")
    print(f"Features saved to {FEATURES_PATH}")
    print(f"Train/test split saved to {SPLIT_PATH}")


if __name__ == "__main__":
    main()