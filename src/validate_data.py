import json

from data_utils import TARGET, build_customer_features, load_transactions
from pipeline_utils import ARTIFACTS_DIR, ensure_project_directories


def main() -> None:
    print("Validating Online Retail data...")
    ensure_project_directories()
    transactions = load_transactions()
    features, cutoff_date = build_customer_features(transactions)
    report = {
        "schema_valid": True,
        "cleaned_transactions": int(len(transactions)),
        "customer_count": int(len(features)),
        "cutoff_date": cutoff_date.isoformat(),
        "target": TARGET,
        "target_counts": {
            str(label): int(count)
            for label, count in features[TARGET].value_counts().sort_index().items()
        },
    }
    (ARTIFACTS_DIR / "data_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"PASS: schema valid; {len(transactions):,} valid transactions; {len(features):,} customers.")


if __name__ == "__main__":
    main()