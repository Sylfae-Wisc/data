"""评估指标：Brier / LogLoss / AUC / F1 / McNemar"""

from pathlib import Path
from typing import Optional

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
    pred_a_bin = (pred_a >= 0.5).astype(int)
    pred_b_bin = (pred_b >= 0.5).astype(int)

    # 列联表
    b = np.sum((pred_a_bin == 1) & (pred_b_bin == 0))  # A 对 B 错
    c = np.sum((pred_a_bin == 0) & (pred_b_bin == 1))  # A 错 B 对

    # McNemar 统计量
    numerator = (b - c) ** 2
    denominator = b + c
    if denominator == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    statistic = numerator / denominator
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def regression_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Logistic 校准评估：预测概率 vs 实际频率"""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    # 计算 ECE（Expected Calibration Error）
    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for i in range(10):
        mask = bin_ids == i
        if mask.sum() > 0:
            ece += np.abs(prob_true[i] - prob_pred[i]) * (mask.sum() / len(y_prob))

    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "ece": float(ece),
    }


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "",
) -> dict:
    """全面评估一个模型的预测"""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "model": model_name,
        "brier": brier(y_true, y_prob),
        "log_loss": calc_log_loss(y_true, y_prob),
        "roc_auc": calc_roc_auc(y_true, y_prob),
        "accuracy": calc_accuracy(y_true, y_pred),
        "f1": calc_f1(y_true, y_pred),
    }
