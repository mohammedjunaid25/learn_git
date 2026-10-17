"""Part B: train, evaluate, and persist a fraud triage model."""
from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("transactions_clean.csv")
MODEL_PATH = Path("fraud_model.pkl")
THRESHOLD_PATH = Path("alert_threshold.txt")
RANDOM_STATE = 2605
CATEGORICAL = ["merchant_category", "city", "payment_mode"]
NUMERIC = ["amount_inr", "hour_of_day", "txn_count_last_24h", "avg_amount_30d_inr",
           "distance_from_home_km", "is_new_device", "account_age_days", "amount_to_avg_ratio",
           "amount_log1p", "avg_amount_log1p", "distance_log1p", "is_night", "high_velocity", "new_account"]
FEATURES = CATEGORICAL + NUMERIC


def _pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC),
    ])
    return Pipeline([("preprocess", preprocessor),
                     ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))])


def _choose_threshold(y_true, probabilities):
    """Choose the lowest threshold meeting a validation precision floor, maximizing recall."""
    candidates = sorted(set([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]))
    valid = []
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        if precision >= 0.30:
            valid.append((recall, precision, threshold))
    if not valid:
        return 0.50
    # Highest recall, then precision, then prefer the higher threshold to limit alert volume.
    return float(max(valid, key=lambda item: (item[0], item[1], item[2]))[2])


def build_final_model():
    """Train on train+validation; choose threshold on validation and report untouched test metrics."""
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df["is_fraud"].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)

    evaluation_model = _pipeline()
    evaluation_model.fit(X_train, y_train)
    val_prob = evaluation_model.predict_proba(X_val)[:, 1]
    threshold = _choose_threshold(y_val, val_prob)
    test_prob = evaluation_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    final_model = _pipeline()
    final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    with MODEL_PATH.open("wb") as f:
        pickle.dump({"model": final_model, "features": FEATURES, "threshold": threshold}, f)
    THRESHOLD_PATH.write_text(f"{threshold:.4f}\n", encoding="utf-8")

    print("PART B SPLITS")
    print(f"total_rows={len(df)}")
    print(f"train_rows={len(X_train)} train_fraud={int(y_train.sum())}")
    print(f"validation_rows={len(X_val)} validation_fraud={int(y_val.sum())}")
    print(f"test_rows={len(X_test)} test_fraud={int(y_test.sum())}")
    print("MODEL: LogisticRegression(class_weight='balanced') with one-hot categories and scaled numeric features")
    print("THRESHOLD SELECTION: maximize validation recall among thresholds with precision >= 0.30")
    print(f"alert_threshold={threshold:.4f}")
    print("TEST METRICS")
    print(f"average_precision={average_precision_score(y_test, test_prob):.4f}")
    print(f"roc_auc={roc_auc_score(y_test, test_prob):.4f}")
    print(f"precision={precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"recall={recall_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"f1={f1_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"confusion_matrix_tn_fp_fn_tp={tn}_{fp}_{fn}_{tp}")
    print(f"saved_model={MODEL_PATH}")
    return final_model, threshold


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_final_model()
