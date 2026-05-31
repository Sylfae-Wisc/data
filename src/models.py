"""模型训练：XGBoost + Walk-forward CV"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")

# Walk-forward 时间窗口（按 Year）
WF_SPLITS = [
    ("2021-2022", "2023"),
    ("2021-2023", "2024"),
    ("2021-2024", "2025"),
    ("2021-2025", "2026"),
]


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_DIR / "match_features.parquet")
    return df


def get_feature_columns() -> list[str]:
    return [
        # Form features
        "team_a_form_5", "team_b_form_5", "form_diff_5",
        "team_a_form_10", "team_b_form_10", "form_diff_10",
        "team_a_form_20", "team_b_form_20", "form_diff_20",
        # H2H features
        "h2h_ratio",
        # Diff features
        "diff_acs", "diff_kast", "diff_adr", "diff_hs_pct",
        "diff_rating", "diff_kd", "diff_fk",
    ]


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def run_walk_forward_cv(df: pd.DataFrame, features: list[str]):
    """Walk-forward CV 训练与评估"""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score, f1_score

    results = []

    for train_label, test_year in WF_SPLITS:
        train_years = [int(y) for y in train_label.split("-")]
        train_mask = df["Year"].isin(train_years)
        test_mask = df["Year"] == int(test_year)

        X_train = df.loc[train_mask, features].values
        y_train = df.loc[train_mask, "target"].values
        X_test = df.loc[test_mask, features].values
        y_test = df.loc[test_mask, "target"].values

        if len(X_test) == 0:
            continue

        # Impute NaN（特征中可能有空值）
        imp = SimpleImputer(strategy="mean")
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

        # 训练
        xgb = train_xgboost(X_train, y_train)
        lr = train_logistic_regression(X_train, y_train)

        # 预测
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        lr_proba = lr.predict_proba(X_test)[:, 1]

        # 评估
        for name, proba in [("XGBoost", xgb_proba), ("Logistic", lr_proba)]:
            results.append({
                "train": train_label,
                "test": str(test_year),
                "model": name,
                "brier": brier_score_loss(y_test, proba),
                "log_loss": log_loss(y_test, proba),
                "roc_auc": roc_auc_score(y_test, proba),
                "accuracy": accuracy_score(y_test, (proba >= 0.5).astype(int)),
                "f1": f1_score(y_test, (proba >= 0.5).astype(int)),
                "n_train": len(y_train),
                "n_test": len(y_test),
            })
            print(f"  [{name}] Train={train_label} Test={test_year}  "
                  f"Brier={results[-1]['brier']:.4f}  AUC={results[-1]['roc_auc']:.4f}  "
                  f"Acc={results[-1]['accuracy']:.4f}")

    return pd.DataFrame(results)


def train_final_model(df: pd.DataFrame, features: list[str]):
    """用全部历史数据训练最终模型"""
    X = df[features].values
    y = df["target"].values

    imp = SimpleImputer(strategy="mean")
    X = imp.fit_transform(X)

    xgb = train_xgboost(X, y)
    joblib.dump(xgb, MODELS_DIR / "xgb_model.pkl")
    joblib.dump(imp, MODELS_DIR / "imputer.pkl")

    # 特征重要性
    importance = pd.DataFrame({
        "feature": features,
        "importance": xgb.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    return xgb


def run_training():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    df = load_features()
    features = get_feature_columns()
    print(f"Loaded {len(df)} samples, {len(features)} features")

    # 检查空值
    null_counts = df[features].isnull().sum()
    if null_counts.sum() > 0:
        print(f"Columns with NaN: {null_counts[null_counts > 0].to_dict()}")

    print("\n=== Walk-forward CV ===")
    cv_results = run_walk_forward_cv(df, features)
    cv_results.to_csv(MODELS_DIR / "cv_results.csv", index=False)

    # 汇总
    print("\n=== CV Summary ===")
    summary = cv_results.groupby("model")[["brier", "roc_auc", "accuracy", "f1"]].mean()
    print(summary.to_string())

    print("\n=== Training final model ===")
    train_final_model(df, features)

    print("\nDone. Models saved to models/")


if __name__ == "__main__":
    run_training()
