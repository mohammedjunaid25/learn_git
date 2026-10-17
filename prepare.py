"""Part A: audit, clean, and engineer synthetic transaction data."""
from pathlib import Path
import pandas as pd

RAW_PATH = Path("transactions_raw.csv")
CLEAN_PATH = Path("transactions_clean.csv")
AMOUNT_MIN = 0.0
AMOUNT_MAX = 1_000_000.0


def clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return one trusted row per transaction; do not mutate the caller's frame."""
    df = raw_df.copy()
    df = df.drop_duplicates(subset=["txn_id"], keep="first").copy()

    for col in ("merchant_category", "city", "payment_mode"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
            df[col] = df[col].str.title()

    numeric_cols = [
        "amount_inr", "hour_of_day", "txn_count_last_24h", "avg_amount_30d_inr",
        "distance_from_home_km", "is_new_device", "account_age_days",
        "chargeback_filed", "is_fraud",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Amount is required for triage. Missing, non-positive, and implausibly high
    # amounts are data-entry failures, so remove them instead of guessing a value.
    df = df[df["amount_inr"].notna()].copy()
    df = df[df["amount_inr"].between(AMOUNT_MIN, AMOUNT_MAX, inclusive="right")].copy()
    df = df[df["avg_amount_30d_inr"].notna() & (df["avg_amount_30d_inr"] > 0)].copy()

    # Distance is allowed to be missing; median imputation is fitted on this table.
    distance_median = df["distance_from_home_km"].median()
    df["distance_from_home_km"] = df["distance_from_home_km"].fillna(distance_median)
    df["is_fraud"] = df["is_fraud"].astype("int64")
    df["hour_of_day"] = df["hour_of_day"].astype("int64")
    df["txn_count_last_24h"] = df["txn_count_last_24h"].astype("int64")
    df["is_new_device"] = df["is_new_device"].astype("int64")
    df["account_age_days"] = df["account_age_days"].astype("int64")
    df["chargeback_filed"] = df["chargeback_filed"].astype("int64")
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic risk features; exclude the outcome and post-event chargeback."""
    out = df.copy()
    out["amount_to_avg_ratio"] = out["amount_inr"] / out["avg_amount_30d_inr"].clip(lower=1.0)
    out["amount_log1p"] = (out["amount_inr"].clip(lower=0)).map(lambda x: __import__("math").log1p(x))
    out["avg_amount_log1p"] = out["avg_amount_30d_inr"].clip(lower=0).map(lambda x: __import__("math").log1p(x))
    out["distance_log1p"] = out["distance_from_home_km"].clip(lower=0).map(lambda x: __import__("math").log1p(x))
    out["is_night"] = out["hour_of_day"].between(0, 5).astype("int64")
    out["high_velocity"] = (out["txn_count_last_24h"] > 6).astype("int64")
    out["new_account"] = (out["account_age_days"] < 30).astype("int64")
    return out


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    cleaned = clean_transactions(raw)
    featured = add_features(cleaned)
    featured.to_csv(CLEAN_PATH, index=False)
    print("PART A AUDIT")
    print(f"raw_rows={len(raw)}")
    print(f"unique_txn_ids={raw['txn_id'].nunique()}")
    print(f"duplicate_rows_removed={len(raw) - raw['txn_id'].nunique()}")
    print(f"missing_amount_rows={pd.to_numeric(raw.drop_duplicates('txn_id')['amount_inr'], errors='coerce').isna().sum()}")
    amount = pd.to_numeric(raw.drop_duplicates("txn_id")["amount_inr"], errors="coerce")
    invalid = amount.notna() & ((amount <= AMOUNT_MIN) | (amount > AMOUNT_MAX))
    print(f"invalid_amount_rows={int(invalid.sum())}")
    print(f"clean_rows={len(featured)}")
    print(f"fraud_rows={int(featured['is_fraud'].sum())}")
    print(f"fraud_rate_pct={featured['is_fraud'].mean() * 100:.2f}")
    print(f"missing_distance_after_cleaning={int(featured['distance_from_home_km'].isna().sum())}")
    print(f"feature_columns_added=amount_to_avg_ratio,amount_log1p,avg_amount_log1p,distance_log1p,is_night,high_velocity,new_account")
    print(f"saved={CLEAN_PATH}")


if __name__ == "__main__":
    main()
