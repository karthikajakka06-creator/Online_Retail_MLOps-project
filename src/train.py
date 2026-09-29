import joblib
from sklearn.ensemble import RandomForestClassifier

from pipeline_utils import BASELINE_MODEL_PATH, ensure_project_directories, load_split


def main() -> None:
    print("Starting baseline Random Forest training...")
    ensure_project_directories()
    split = load_split()
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(split["X_train"], split["y_train"])
    joblib.dump(model, BASELINE_MODEL_PATH)
    print(f"Baseline model saved to {BASELINE_MODEL_PATH}")


if __name__ == "__main__":
    main()