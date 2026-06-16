"""评估指标：Brier / LogLoss / AUC / F1 / McNemar"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from scipy.stats import chi2

PREDICTION_THRESHOLD = 0.5
SIGNIFICANCE_LEVEL = 0.05
CALIBRATION_BIN_COUNT = 10

EVALUATION_METRICS = [
    ("brier", "probability", brier_score_loss),
    ("log_loss", "probability", log_loss),
    ("roc_auc", "probability", roc_auc_score),
    ("accuracy", "class", accuracy_score),
    ("f1", "class", f1_score),
]


def _to_binary_predictions(
    values: np.ndarray,
    threshold: float = PREDICTION_THRESHOLD,
) -> np.ndarray:
    return (values >= threshold).astype(int)


def _mcnemar_disagreements(
    pred_a_bin: np.ndarray,
    pred_b_bin: np.ndarray,
) -> tuple[int, int]:
    # 列联表
    b = np.sum((pred_a_bin == 1) & (pred_b_bin == 0))  # A 对 B 错
    c = np.sum((pred_a_bin == 0) & (pred_b_bin == 1))  # A 错 B 对
    return int(b), int(c)


def _mcnemar_statistic(b: int, c: int) -> float:
    denominator = b + c
    if denominator == 0:
        return 0.0
    return ((b - c) ** 2) / denominator


def _mcnemar_p_value(statistic: float) -> float:
    return 1 - chi2.cdf(statistic, df=1)


def _calibration_bins() -> np.ndarray:
    return np.linspace(0, 1, CALIBRATION_BIN_COUNT + 1)


def _expected_calibration_error(
    y_prob: np.ndarray,
    prob_true: np.ndarray,
    prob_pred: np.ndarray,
) -> float:
    bins = _calibration_bins()
    bin_ids = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for i in range(CALIBRATION_BIN_COUNT):
        mask = bin_ids == i
        if mask.sum() > 0:
            ece += np.abs(prob_true[i] - prob_pred[i]) * (mask.sum() / len(y_prob))
    return float(ece)


def _metric_value(
    metric_kind: str,
    metric_func,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    if metric_kind == "probability":
        return metric_func(y_true, y_prob)
    if metric_kind == "class":
        return metric_func(y_true, y_pred)
    raise ValueError(f"Unsupported metric kind: {metric_kind}")


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return brier_score_loss(y_true, y_prob)


def calc_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return log_loss(y_true, y_prob)


def calc_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return roc_auc_score(y_true, y_prob)


def calc_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def calc_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred)


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict:
    """McNemar's Test：比较两个模型的预测是否显著不同"""
    pred_a_bin = _to_binary_predictions(pred_a)
    pred_b_bin = _to_binary_predictions(pred_b)
    b, c = _mcnemar_disagreements(pred_a_bin, pred_b_bin)

    statistic = _mcnemar_statistic(b, c)
    if statistic == 0.0 and b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    p_value = _mcnemar_p_value(statistic)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < SIGNIFICANCE_LEVEL),
    }


def regression_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Logistic 校准评估：预测概率 vs 实际频率"""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=CALIBRATION_BIN_COUNT,
    )
    # 计算 ECE（Expected Calibration Error）
    ece = _expected_calibration_error(y_prob, prob_true, prob_pred)

    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "ece": ece,
    }


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "",
) -> dict:
    """全面评估一个模型的预测"""
    y_pred = _to_binary_predictions(y_prob)
    results = {
        "model": model_name,
    }
    for metric_name, metric_kind, metric_func in EVALUATION_METRICS:
        results[metric_name] = _metric_value(
            metric_kind,
            metric_func,
            y_true,
            y_prob,
            y_pred,
        )
    return results
