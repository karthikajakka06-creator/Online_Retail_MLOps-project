import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from pipeline_utils import ARTIFACTS_DIR, ensure_project_directories, load_split


def main() -> None:
    print("Running reproducibility validation...")
    ensure_project_directories()
    split = load_split()
    runs = []
    for _ in range(2):
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(split["X_train"], split["y_train"])
        runs.append(
            {
                "predictions": model.predict(split["X_test"]),
                "f1": float(f1_score(split["y_test"], model.predict(split["X_test"]))),
            }
        )

    reproducible = np.array_equal(runs[0]["predictions"], runs[1]["predictions"])
    reproducible = reproducible and runs[0]["f1"] == runs[1]["f1"]
    report = {
        "execution_1_f1": runs[0]["f1"],
        "execution_2_f1": runs[1]["f1"],
        "predictions_identical": bool(np.array_equal(runs[0]["predictions"], runs[1]["predictions"])),
        "reproducible": bool(reproducible),
    }
    (ARTIFACTS_DIR / "reproducibility_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not reproducible:
        raise RuntimeError("Repeated training produced different predictions")
    print(f"PASS: repeated F1={runs[0]['f1']:.6f}; predictions are identical.")


if __name__ == "__main__":
    main()