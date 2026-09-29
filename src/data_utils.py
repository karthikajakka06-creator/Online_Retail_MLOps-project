from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "OnlineRetail.csv"
FEATURES_PATH = PROJECT_ROOT / "outputs" / "customer_features.csv"
TARGET = "FuturePurchase"
FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "UniqueProducts",
    "AvgTransactionValue",
]
REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}


def load_transactions(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    transactions = pd.read_csv(path, encoding="ISO-8859-1")
    missing_columns = REQUIRED_COLUMNS.difference(transactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    transactions = transactions.dropna(subset=["CustomerID", "Description"])
    transactions = transactions.drop_duplicates().copy()
    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"], errors="coerce"
    )
    if transactions["InvoiceDate"].isna().any():
        raise ValueError("Dataset contains invalid InvoiceDate values")

    transactions = transactions.loc[
        (transactions["Quantity"] > 0) & (transactions["UnitPrice"] > 0)
    ].copy()
    transactions["TotalAmount"] = transactions["Quantity"] * transactions["UnitPrice"]
    if transactions.empty:
        raise ValueError("No valid purchase transactions remain after cleaning")

    return transactions


def build_customer_features(
    transactions: pd.DataFrame, forecast_days: int = 30
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if forecast_days < 1:
        raise ValueError("forecast_days must be at least 1")

    cutoff_date = transactions["InvoiceDate"].max() - pd.Timedelta(days=forecast_days)
    historical = transactions.loc[transactions["InvoiceDate"] <= cutoff_date]
    future = transactions.loc[transactions["InvoiceDate"] > cutoff_date]
    if historical.empty or future.empty:
        raise ValueError("Historical and future periods must both contain transactions")

    customer_features = historical.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (cutoff_date - dates.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
        TotalQuantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        AvgTransactionValue=("TotalAmount", "mean"),
    ).reset_index()

    future_customers = set(future["CustomerID"].unique())
    customer_features[TARGET] = customer_features["CustomerID"].isin(
        future_customers
    ).astype("int8")

    if customer_features.empty or customer_features[TARGET].nunique() != 2:
        raise ValueError("Customer dataset must contain both target classes")
    if not customer_features[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).notna().all().all():
        raise ValueError("Customer features contain missing or non-numeric values")

    return customer_features, cutoff_date