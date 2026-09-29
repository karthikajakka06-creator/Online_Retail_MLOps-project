# Online Retail MLOps Project

This project predicts whether an existing Online Retail customer will purchase during the final 30 days of the dataset. Historical behavior is aggregated into six features, matching `online_retail_ml.ipynb`; customer IDs are excluded from model inputs.

## Project Structure

```text
Online_Retail_MLops-project/
|-- data/            # OnlineRetail.csv
|-- notebooks/       # Optional notebooks
|-- src/             # Preprocessing, training, validation, registry scripts
|-- pipelines/       # Lab 3-6 runners and run_all.py
|-- models/          # Trained model artifacts
|-- outputs/         # Feature, evaluation, and transformed datasets
|-- logs/            # Reserved for runtime logs
|-- artifacts/       # Split, reports, and local MLflow database
|-- online_retail_ml.ipynb
|-- requirements.txt
`-- README.md
```

## Step 1: Create the project structure

The folders and scripts are laid out above. The source dataset belongs at `data/OnlineRetail.csv`.

## Step 2: Place the dataset

Keep the supplied `OnlineRetail.csv` in `data/`. The loader uses the dataset's ISO-8859-1 encoding, removes rows missing customer or product descriptions, removes duplicate rows, then excludes returns and non-positive prices.

## Step 3: Preprocess

`src/preprocess.py` builds the historical customer features and the 30-day `FuturePurchase` label, then saves a stratified, fixed-seed train/test split.

## Step 4: Train

`src/train.py` trains the baseline Random Forest and saves it under `models/`.

## Step 5: Evaluate

`src/evaluate.py` writes accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, and misclassified customer rows under `outputs/`.

## Step 6: Run Lab 3

```powershell
python .\pipelines\run_lab3_baseline.py
```

## Step 7: Add MLflow tracking, Lab 4

```powershell
python .\pipelines\run_lab4_tracking.py
```

MLflow uses a local SQLite tracking and registry database at `artifacts/mlflow.db`; no hosted tracking server is required. Runs and logged model artifacts are stored locally.

## Step 8: Run the production pipeline, Lab 5

```powershell
python .\pipelines\run_lab5_pipeline.py
```

This validates the source data, fits a scikit-learn preprocessing/model pipeline, and checks output dimensions and values.

## Step 9: Register and promote, Lab 6

```powershell
python .\pipelines\run_lab6_registry.py
```

Each tracked model is registered as a `candidate`; it becomes `champion` only if its held-out F1 is at least as high as the current champion. MLflow aliases are used because model stages are deprecated.

## Step 10: Execute everything

Create and activate a virtual environment, install dependencies, then run all labs:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\pipelines\run_all.py
```

## Step 11: Publish to GitHub

Create an empty GitHub repository, then from this project folder:

```powershell
git init
git add README.md requirements.txt .gitignore data src pipelines online_retail_ml.ipynb
git commit -m "Build Online Retail MLOps pipeline"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repository>.git
git push -u origin main
```

The local MLflow database and generated caches are excluded from Git. Review dataset redistribution terms before publishing the CSV.