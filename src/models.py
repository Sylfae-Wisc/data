"""模型训练：XGBoost + Walk-forward CV"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")

# Walk-forward 时间窗口（按 Year）
WF_SPLITS = [
    ("2021-2022", "2023"),
    ("2021-2023", "2024"),
    ("2021-2024", "2025"),
    ("2021-2025", "2026"),
]

FORM_FEATURE_COLUMNS = [
    "team_a_form_5",
    "team_b_form_5",
    "form_diff_5",
    "team_a_form_10",
    "team_b_form_10",
    "form_diff_10",
    "team_a_form_20",
    "team_b_form_20",
    "form_diff_20",
]

H2H_FEATURE_COLUMNS = [
    "h2h_ratio",
]

DIFF_FEATURE_COLUMNS = [
    "diff_acs",
    "diff_kast",
    "diff_adr",
    "diff_hs_pct",
    "diff_rating",
    "diff_kd",
    "diff_fk",
]

CV_METRIC_COLUMNS = [
    "brier",
    "roc_auc",
    "accuracy",
    "f1",
]

MODEL_FACTORIES = [
    ("XGBoost", "xgb"),
    ("Logistic", "logistic"),
]


def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_DIR / "match_features.parquet")


def get_feature_columns() -> list[str]:
    return FORM_FEATURE_COLUMNS + H2H_FEATURE_COLUMNS + DIFF_FEATURE_COLUMNS


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


def _parse_train_years(train_label: str) -> list[int]:
    return [int(year) for year in train_label.split("-")]


def _split_walk_forward_data(
    df: pd.DataFrame,
    features: list[str],
    train_label: str,
    test_year: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_years = _parse_train_years(train_label)
    train_mask = df["Year"].isin(train_years)
    test_mask = df["Year"] == int(test_year)

    X_train = df.loc[train_mask, features].values
    y_train = df.loc[train_mask, "target"].values
    X_test = df.loc[test_mask, features].values
    y_test = df.loc[test_mask, "target"].values

    return X_train, y_train, X_test, y_test


def _impute_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    imp = SimpleImputer(strategy="mean")
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)
    return X_train, X_test, imp


def _train_named_model(
    model_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    if model_key == "xgb":
        return train_xgboost(X_train, y_train)
    if model_key == "logistic":
        return train_logistic_regression(X_train, y_train)
    raise ValueError(f"Unsupported model key: {model_key}")


def _classification_metrics(
    y_test: np.ndarray,
    proba: np.ndarray,
) -> dict:
    predictions = (proba >= 0.5).astype(int)
    return {
        "brier": brier_score_loss(y_test, proba),
        "log_loss": log_loss(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }


def _cv_result_row(
    train_label: str,
    test_year: str,
    model_name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    proba: np.ndarray,
) -> dict:
    metrics = _classification_metrics(y_test, proba)
    return {
        "train": train_label,
        "test": str(test_year),
        "model": model_name,
        **metrics,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def _print_cv_result(row: dict) -> None:
    print(
        f"  [{row['model']}] Train={row['train']} Test={row['test']}  "
        f"Brier={row['brier']:.4f}  AUC={row['roc_auc']:.4f}  "
        f"Acc={row['accuracy']:.4f}"
    )


def _fit_imputer(X: np.ndarray) -> tuple[np.ndarray, SimpleImputer]:
    imp = SimpleImputer(strategy="mean")
    return imp.fit_transform(X), imp


def _build_importance_frame(
    features: list[str],
    model: XGBClassifier,
) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)


def run_walk_forward_cv(df: pd.DataFrame, features: list[str]):
    """Walk-forward CV 训练与评估"""
    results = []

    for train_label, test_year in WF_SPLITS:
        X_train, y_train, X_test, y_test = _split_walk_forward_data(
            df,
            features,
            train_label,
            test_year,
        )

        if len(X_test) == 0:
            continue

        # Impute NaN（特征中可能有空值）
        X_train, X_test, _ = _impute_train_test(X_train, X_test)

        # 训练
        trained_models = [
            (
                model_name,
                _train_named_model(model_key, X_train, y_train),
            )
            for model_name, model_key in MODEL_FACTORIES
        ]

        # 评估
        for model_name, model in trained_models:
            proba = model.predict_proba(X_test)[:, 1]
            row = _cv_result_row(
                train_label=train_label,
                test_year=test_year,
                model_name=model_name,
                y_train=y_train,
                y_test=y_test,
                proba=proba,
            )
            results.append(row)
            _print_cv_result(row)

    return pd.DataFrame(results)


def train_final_model(df: pd.DataFrame, features: list[str]):
    """用全部历史数据训练最终模型"""
    X = df[features].values
    y = df["target"].values

    X, imp = _fit_imputer(X)

    xgb = train_xgboost(X, y)
    joblib.dump(xgb, MODELS_DIR / "xgb_model.pkl")
    joblib.dump(imp, MODELS_DIR / "imputer.pkl")

    # 特征重要性
    importance = _build_importance_frame(features, xgb)
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
    summary = cv_results.groupby("model")[CV_METRIC_COLUMNS].mean()
    print(summary.to_string())

    print("\n=== Training final model ===")
    train_final_model(df, features)

    print("\nDone. Models saved to models/")


if __name__ == "__main__":
    run_training()
