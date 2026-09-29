import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import FEATURE_COLUMNS
from pipeline_utils import (
    PREPROCESSOR_PATH,
    PRODUCTION_MODEL_PATH,
    OUTPUTS_DIR,
    ensure_project_directories,
    load_split,
)
import joblib


def main() -> None:
    print("Fitting production preprocessing and model pipeline...")
    ensure_project_directories()
    split = load_split()
    preprocessor = ColumnTransformer(
        [("scale", StandardScaler(), FEATURE_COLUMNS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            ),
        ]
    )
    pipeline.fit(split["X_train"], split["y_train"])
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    joblib.dump(fitted_preprocessor, PREPROCESSOR_PATH)
    joblib.dump(pipeline, PRODUCTION_MODEL_PATH)

    for name in ("train", "test"):
        transformed = fitted_preprocessor.transform(split[f"X_{name}"])
        pd.DataFrame(transformed, columns=FEATURE_COLUMNS).to_csv(
            OUTPUTS_DIR / f"preprocessed_{name}.csv", index=False
        )
    print(f"Pipeline saved to {PRODUCTION_MODEL_PATH}")
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()