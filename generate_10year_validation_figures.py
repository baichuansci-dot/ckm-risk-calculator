#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 10-year validation figures for the web risk calculator.

Outputs:
1. IPCW time-dependent ROC curves and Youden cutoffs at 120 months
2. Decision curve analysis (DCA) at 120 months
3. 10-year calibration curves
4. CSV tables and a text summary
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR.parents[1]
DATA_DIR = ARCHIVE_DIR / "01_数据构建_训练测试划分_标准化"
OUTPUT_DIR = BASE_DIR / "validation_outputs_10year"
MODEL_DIR = BASE_DIR / "models"

TRAIN_DATA_PATH = DATA_DIR / "训练集_标准化后.csv"
TEST_DATA_PATH = DATA_DIR / "测试集_标准化后.csv"

HORIZON_MONTHS = 120.0
RANDOM_STATE = 42

ALL_CAUSE_FEATURES = [
    "WBC", "HighCholesterol", "DBP", "Creatinine", "Glucose",
    "Gender", "TG", "SBP", "Age", "MCV", "smoking", "Platelet", "CI",
]

CARDIO_FEATURES = [
    "WBC", "DBP", "UricAcid", "SBP", "BUN", "Age", "CI",
]

TASKS = [
    {
        "key": "all_cause",
        "label": "All-cause mortality",
        "model_name": "GradientBoostingSurvival",
        "model_file": "CI_all_cause_death_GradientBoostingSurvival.pkl",
        "features": ALL_CAUSE_FEATURES,
        "event_builder": "all_cause",
    },
    {
        "key": "cardiovascular",
        "label": "Cardiovascular mortality",
        "model_name": "RandomSurvivalForest",
        "model_file": "CI_cardiovascular_death_RandomSurvivalForest.pkl",
        "features": CARDIO_FEATURES,
        "event_builder": "cardiovascular",
    },
]


def print_step(message: str) -> None:
    print(f"\n【步骤】{message}")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print_step("读取训练集和测试集标准化数据")
    train = pd.read_csv(TRAIN_DATA_PATH)
    test = pd.read_csv(TEST_DATA_PATH)
    print(f"训练集: {TRAIN_DATA_PATH}")
    print(f"训练集样本量: {train.shape[0]}, 字段数: {train.shape[1]}")
    print(f"测试集: {TEST_DATA_PATH}")
    print(f"测试集样本量: {test.shape[0]}, 字段数: {test.shape[1]}")
    return train, test


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["all_cause_event"] = (df["mortstat"] == 1).astype(int)
    df["cardiovascular_event"] = (
        (df["mortstat"] == 1) & (df["ucod_leading"].isin([1, 1.0, 5, 5.0]))
    ).astype(int)
    return df


def predict_risk_at_time(model, X: np.ndarray, time_point: float) -> np.ndarray:
    if not hasattr(model, "predict_survival_function"):
        raise RuntimeError(f"模型 {type(model)} 不支持 predict_survival_function，不能输出固定时间点绝对风险。")
    risks = []
    for surv_func in model.predict_survival_function(X):
        surv_prob = float(surv_func(time_point))
        risks.append(1.0 - surv_prob)
    return np.clip(np.asarray(risks, dtype=float), 0.0, 1.0)


def km_censoring_survival(times: np.ndarray, events: np.ndarray):
    """Kaplan-Meier estimate for censoring survival G(t)=P(C>=t)."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    censor_events = 1 - events
    unique_times = np.sort(np.unique(times[censor_events == 1]))
    surv_values = []
    surv = 1.0
    for t in unique_times:
        at_risk = np.sum(times >= t)
        d = np.sum((times == t) & (censor_events == 1))
        if at_risk > 0:
            surv *= max(0.0, 1.0 - d / at_risk)
        surv_values.append(surv)
    unique_times = np.asarray(unique_times, dtype=float)
    surv_values = np.asarray(surv_values, dtype=float)

    def g(query_time):
        q = np.asarray(query_time, dtype=float)
        if len(unique_times) == 0:
            out = np.ones_like(q, dtype=float)
        else:
            idx = np.searchsorted(unique_times, q, side="right") - 1
            out = np.ones_like(q, dtype=float)
            valid = idx >= 0
            out[valid] = surv_values[idx[valid]]
        return np.maximum(out, 1e-6)

    return g


def ipcw_binary_components(times: np.ndarray, events: np.ndarray, horizon: float):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    g = km_censoring_survival(times, events)

    case_mask = (events == 1) & (times <= horizon)
    control_mask = times > horizon
    early_censored_mask = (events == 0) & (times <= horizon)

    case_weights = np.zeros_like(times, dtype=float)
    control_weights = np.zeros_like(times, dtype=float)
    case_weights[case_mask] = 1.0 / g(times[case_mask])
    control_weights[control_mask] = 1.0 / float(g(np.array([horizon]))[0])

    return case_mask, control_mask, early_censored_mask, case_weights, control_weights


def calculate_ipcw_roc_youden(times: np.ndarray, events: np.ndarray, risks: np.ndarray, horizon: float):
    case_mask, control_mask, early_censored_mask, case_w, control_w = ipcw_binary_components(times, events, horizon)
    total_case_w = case_w.sum()
    total_control_w = control_w.sum()
    if total_case_w <= 0 or total_control_w <= 0:
        raise RuntimeError("IPCW ROC无法计算：病例或对照权重为0。")

    thresholds = np.unique(np.r_[0.0, np.sort(risks), 1.0])
    rows = []
    for threshold in thresholds:
        positive = risks >= threshold
        sensitivity = case_w[positive].sum() / total_case_w
        specificity = control_w[~positive].sum() / total_control_w
        fpr = 1.0 - specificity
        youden = sensitivity + specificity - 1.0
        rows.append({
            "threshold": float(threshold),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "youden": float(youden),
        })
    roc_df = pd.DataFrame(rows).sort_values(["fpr", "sensitivity"]).drop_duplicates("fpr", keep="last")
    roc_df = roc_df.sort_values("fpr")
    auc = float(np.trapz(roc_df["sensitivity"].values, roc_df["fpr"].values))
    full_df = pd.DataFrame(rows)
    best_idx = full_df["youden"].idxmax()
    best = full_df.loc[best_idx].to_dict()
    best["auc_ipcw"] = auc
    best["n_events_before_10y"] = int(case_mask.sum())
    best["n_controls_after_10y"] = int(control_mask.sum())
    best["n_censored_before_10y"] = int(early_censored_mask.sum())
    return roc_df, full_df, best


def calculate_metrics_at_threshold(times: np.ndarray, events: np.ndarray, risks: np.ndarray, horizon: float, threshold: float):
    """Calculate IPCW sensitivity/specificity at a prespecified threshold."""
    _, _, _, case_w, control_w = ipcw_binary_components(times, events, horizon)
    total_case_w = case_w.sum()
    total_control_w = control_w.sum()
    positive = risks >= threshold
    sensitivity = case_w[positive].sum() / total_case_w if total_case_w > 0 else np.nan
    specificity = control_w[~positive].sum() / total_control_w if total_control_w > 0 else np.nan
    youden = sensitivity + specificity - 1.0
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "fpr": float(1.0 - specificity),
        "youden": float(youden),
    }


def calculate_dca_ipcw(times: np.ndarray, events: np.ndarray, risks: np.ndarray, horizon: float):
    case_mask, control_mask, _, case_w, control_w = ipcw_binary_components(times, events, horizon)
    total_w = case_w.sum() + control_w.sum()
    prevalence = case_w.sum() / total_w
    thresholds = np.arange(0.001, 0.301, 0.001)
    rows = []
    for pt in thresholds:
        positive = risks >= pt
        tp_w = case_w[positive].sum()
        fp_w = control_w[positive].sum()
        odds = pt / (1.0 - pt)
        model_nb = (tp_w / total_w) - (fp_w / total_w) * odds
        all_nb = prevalence - (1.0 - prevalence) * odds
        rows.append({
            "threshold_probability": float(pt),
            "model_net_benefit": float(model_nb),
            "treat_all_net_benefit": float(all_nb),
            "treat_none_net_benefit": 0.0,
        })
    return pd.DataFrame(rows)


def calculate_calibration(times: np.ndarray, events: np.ndarray, risks: np.ndarray, horizon: float, n_bins: int = 10):
    case_mask, control_mask, _, case_w, control_w = ipcw_binary_components(times, events, horizon)
    valid = case_mask | control_mask
    y = case_mask[valid].astype(int)
    pred = risks[valid]
    sample_w = case_w[valid] + control_w[valid]

    order = np.argsort(pred)
    pred_sorted = pred[order]
    y_sorted = y[order]
    w_sorted = sample_w[order]
    bins = np.array_split(np.arange(len(pred_sorted)), n_bins)

    rows = []
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        w = w_sorted[idx]
        rows.append({
            "bin": i,
            "n": int(len(idx)),
            "mean_predicted_risk": float(np.average(pred_sorted[idx], weights=w)),
            "observed_event_rate_ipcw": float(np.average(y_sorted[idx], weights=w)),
            "min_predicted_risk": float(pred_sorted[idx].min()),
            "max_predicted_risk": float(pred_sorted[idx].max()),
        })
    cal_df = pd.DataFrame(rows)

    brier = float(np.average((y - pred) ** 2, weights=sample_w))
    eps = 1e-6
    logit_pred = np.log(np.clip(pred, eps, 1 - eps) / (1 - np.clip(pred, eps, 1 - eps)))
    try:
        slope = float(np.polyfit(logit_pred, y, deg=1, w=sample_w)[0])
    except Exception:
        slope = np.nan
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(pred, y, sample_weight=sample_w)
    except Exception:
        pass
    return cal_df, brier, slope


def save_roc_plot(roc_df: pd.DataFrame, best: dict, task: dict):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc_df["fpr"], roc_df["sensitivity"], color="#1f77b4", lw=2.2,
            label=f"Model (AUC={best['auc_ipcw']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.2, label="Reference")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"10-year IPCW time-dependent ROC\n{task['label']}")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUTPUT_DIR / f"ROC_10year_{task['key']}_IPCW.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_dca_plot(dca_df: pd.DataFrame, best: dict, task: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = dca_df["threshold_probability"] * 100
    ax.plot(x, dca_df["model_net_benefit"], color="#d62728", lw=2.2, label="Model")
    ax.plot(x, dca_df["treat_all_net_benefit"], color="#1f77b4", lw=1.8, linestyle="--", label="Treat all")
    ax.plot(x, dca_df["treat_none_net_benefit"], color="black", lw=1.5, linestyle=":", label="Treat none")
    ax.axvline(best["threshold"] * 100, color="gray", lw=1.2, linestyle="--", label="Derivation Youden cutoff")
    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"10-year Decision Curve Analysis\n{task['label']}")
    ax.set_xlim(0, 30)

    # Treat-all may become strongly negative at high threshold probabilities,
    # especially for low-incidence cardiovascular mortality. Cropping the lower
    # y-axis improves visibility of the clinically relevant positive net-benefit region.
    positive_region = dca_df[dca_df["threshold_probability"] <= 0.10]
    y_upper = max(positive_region["model_net_benefit"].max(), positive_region["treat_all_net_benefit"].max(), 0.01)
    ax.set_ylim(-0.02, y_upper + 0.01)

    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUTPUT_DIR / f"DCA_10year_{task['key']}_IPCW.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_calibration_plot(cal_df: pd.DataFrame, brier: float, task: dict):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.2, label="Ideal")
    ax.plot(cal_df["mean_predicted_risk"], cal_df["observed_event_rate_ipcw"],
            marker="o", color="#2ca02c", lw=2.0, label=f"Model (Brier={brier:.3f})")
    ax.set_xlabel("Mean predicted 10-year risk")
    ax.set_ylabel("Observed 10-year event rate (IPCW)")
    ax.set_title(f"10-year Calibration Curve\n{task['label']}")
    upper = max(0.1, float(max(cal_df["mean_predicted_risk"].max(), cal_df["observed_event_rate_ipcw"].max())) * 1.2)
    upper = min(1.0, upper)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUTPUT_DIR / f"Calibration_10year_{task['key']}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def analyze_task(train: pd.DataFrame, test: pd.DataFrame, task: dict) -> dict:
    print_step(f"分析任务：{task['label']}")
    model_path = MODEL_DIR / task["model_file"]
    model = joblib.load(model_path)
    print(f"加载模型: {model_path}")

    for data_name, data in [("训练集", train), ("测试集", test)]:
        missing = [c for c in task["features"] if c not in data.columns]
        if missing:
            raise RuntimeError(f"{data_name}缺少模型特征: {missing}")

    event_col = "all_cause_event" if task["event_builder"] == "all_cause" else "cardiovascular_event"

    # 1) Derivation cohort: derive IPCW Youden cutoff from training data.
    X_train = train[task["features"]].values
    train_risks = predict_risk_at_time(model, X_train, HORIZON_MONTHS)
    train_times = train["permth_exm"].values.astype(float)
    train_events = train[event_col].values.astype(int)

    print("【阈值推导】使用训练集推导IPCW Youden cutoff")
    print(f"训练集10年预测风险范围: {train_risks.min():.4f} - {train_risks.max():.4f}，中位数: {np.median(train_risks):.4f}")
    print(f"训练集总样本: {len(train)}")
    print(f"训练集10年内事件数: {int(((train_events == 1) & (train_times <= HORIZON_MONTHS)).sum())}")
    print(f"训练集随访超过10年对照数: {int((train_times > HORIZON_MONTHS).sum())}")
    print(f"训练集10年前删失数: {int(((train_events == 0) & (train_times <= HORIZON_MONTHS)).sum())}")

    train_roc_df, train_roc_full_df, train_best = calculate_ipcw_roc_youden(
        train_times, train_events, train_risks, HORIZON_MONTHS
    )

    # 2) Validation cohort: use test data for ROC/DCA/calibration figures.
    X_test = test[task["features"]].values
    test_risks = predict_risk_at_time(model, X_test, HORIZON_MONTHS)
    test_times = test["permth_exm"].values.astype(float)
    test_events = test[event_col].values.astype(int)

    print("【模型验证】使用测试集生成ROC、DCA和校准曲线")
    print(f"测试集10年预测风险范围: {test_risks.min():.4f} - {test_risks.max():.4f}，中位数: {np.median(test_risks):.4f}")
    print(f"测试集总样本: {len(test)}")
    print(f"测试集10年内事件数: {int(((test_events == 1) & (test_times <= HORIZON_MONTHS)).sum())}")
    print(f"测试集随访超过10年对照数: {int((test_times > HORIZON_MONTHS).sum())}")
    print(f"测试集10年前删失数: {int(((test_events == 0) & (test_times <= HORIZON_MONTHS)).sum())}")

    test_roc_df, test_roc_full_df, test_best = calculate_ipcw_roc_youden(
        test_times, test_events, test_risks, HORIZON_MONTHS
    )
    # Plot and mark the derivation cutoff at its validation-set sensitivity/specificity.
    derivation_best_for_plot = calculate_metrics_at_threshold(
        test_times, test_events, test_risks, HORIZON_MONTHS, float(train_best["threshold"])
    )
    derivation_best_for_plot["auc_ipcw"] = test_best["auc_ipcw"]
    dca_df = calculate_dca_ipcw(test_times, test_events, test_risks, HORIZON_MONTHS)
    cal_df, brier, slope = calculate_calibration(test_times, test_events, test_risks, HORIZON_MONTHS)

    train_roc_df.to_csv(OUTPUT_DIR / f"roc_points_10year_{task['key']}_IPCW_derivation_train.csv", index=False)
    train_roc_full_df.to_csv(OUTPUT_DIR / f"roc_thresholds_10year_{task['key']}_IPCW_derivation_train.csv", index=False)
    test_roc_df.to_csv(OUTPUT_DIR / f"roc_points_10year_{task['key']}_IPCW_validation_test.csv", index=False)
    test_roc_full_df.to_csv(OUTPUT_DIR / f"roc_thresholds_10year_{task['key']}_IPCW_validation_test.csv", index=False)
    dca_df.to_csv(OUTPUT_DIR / f"dca_net_benefit_10year_{task['key']}.csv", index=False)
    cal_df.to_csv(OUTPUT_DIR / f"calibration_10year_{task['key']}.csv", index=False)

    save_roc_plot(test_roc_df, derivation_best_for_plot, task)
    save_dca_plot(dca_df, derivation_best_for_plot, task)
    save_calibration_plot(cal_df, brier, task)

    summary = {
        "outcome": task["key"],
        "outcome_label": task["label"],
        "model": task["model_name"],
        "horizon_months": HORIZON_MONTHS,
        "derivation_dataset": "training set",
        "validation_dataset": "test set",
        "n_total_derivation": int(len(train)),
        "n_events_before_10y_derivation": int(train_best["n_events_before_10y"]),
        "n_controls_after_10y_derivation": int(train_best["n_controls_after_10y"]),
        "n_censored_before_10y_derivation": int(train_best["n_censored_before_10y"]),
        "auc_ipcw_derivation": float(train_best["auc_ipcw"]),
        "threshold_ipcw": float(train_best["threshold"]),
        "youden_ipcw_derivation": float(train_best["youden"]),
        "sensitivity_ipcw_derivation": float(train_best["sensitivity"]),
        "specificity_ipcw_derivation": float(train_best["specificity"]),
        "n_total_validation": int(len(test)),
        "n_events_before_10y_validation": int(test_best["n_events_before_10y"]),
        "n_controls_after_10y_validation": int(test_best["n_controls_after_10y"]),
        "n_censored_before_10y_validation": int(test_best["n_censored_before_10y"]),
        "auc_ipcw_validation": float(test_best["auc_ipcw"]),
        "sensitivity_validation_at_derivation_cutoff": float(derivation_best_for_plot["sensitivity"]),
        "specificity_validation_at_derivation_cutoff": float(derivation_best_for_plot["specificity"]),
        "youden_validation_at_derivation_cutoff": float(derivation_best_for_plot["youden"]),
        "youden_ipcw_validation_at_own_optimum": float(test_best["youden"]),
        "threshold_ipcw_validation_own_optimum": float(test_best["threshold"]),
        "brier_score_ipcw_validation": float(brier),
        "calibration_slope_approx_validation": float(slope) if not np.isnan(slope) else np.nan,
        "method": "Derive cutoff in training set; validate ROC/DCA/calibration in test set; risk = 1 - S(120 months)",
    }
    print(f"训练集IPCW AUC: {summary['auc_ipcw_derivation']:.4f}")
    print(f"训练集IPCW Youden cutoff: {summary['threshold_ipcw']:.4f} ({summary['threshold_ipcw']:.2%})")
    print(f"训练集Sensitivity: {summary['sensitivity_ipcw_derivation']:.4f}")
    print(f"训练集Specificity: {summary['specificity_ipcw_derivation']:.4f}")
    print(f"测试集IPCW AUC: {summary['auc_ipcw_validation']:.4f}")
    print(f"测试集自身最佳cutoff（仅作对照，不用于网页）: {summary['threshold_ipcw_validation_own_optimum']:.4f} ({summary['threshold_ipcw_validation_own_optimum']:.2%})")
    print(f"测试集IPCW Brier score: {summary['brier_score_ipcw_validation']:.4f}")
    return summary


def write_summary(summaries: list[dict]) -> None:
    thresholds_df = pd.DataFrame(summaries)
    thresholds_path = OUTPUT_DIR / "risk_thresholds_10_year_Youden_IPCW.csv"
    thresholds_df.to_csv(thresholds_path, index=False)

    root_thresholds_path = BASE_DIR / "risk_thresholds_10_year_Youden_IPCW.csv"
    thresholds_df.to_csv(root_thresholds_path, index=False)

    text_path = OUTPUT_DIR / "validation_summary_10year.txt"
    lines = []
    lines.append("10年网页风险计算器验证结果汇总")
    lines.append("=" * 60)
    lines.append(f"风险定义: 10年死亡风险 = 1 - S({HORIZON_MONTHS:.0f} months)")
    lines.append(f"训练集: {TRAIN_DATA_PATH}")
    lines.append(f"测试集: {TEST_DATA_PATH}")
    lines.append("心血管死亡定义: mortstat == 1 且 ucod_leading in {1, 5}")
    lines.append("风险分层建议: 使用训练集推导的IPCW Youden cutoff；预测10年风险 >= cutoff 为高风险，否则为低风险/非高风险。")
    lines.append("测试集仅用于验证ROC、DCA和校准曲线，不用于确定网页阈值。")
    lines.append("")
    for s in summaries:
        lines.append(f"任务: {s['outcome_label']}")
        lines.append(f"模型: {s['model']}")
        lines.append("【阈值推导：训练集】")
        lines.append(f"训练集总样本: {s['n_total_derivation']}")
        lines.append(f"训练集10年内事件数: {s['n_events_before_10y_derivation']}")
        lines.append(f"训练集随访超过10年对照数: {s['n_controls_after_10y_derivation']}")
        lines.append(f"训练集10年前删失数: {s['n_censored_before_10y_derivation']}")
        lines.append(f"训练集IPCW AUC: {s['auc_ipcw_derivation']:.4f}")
        lines.append(f"训练集IPCW Youden cutoff: {s['threshold_ipcw']:.4f} ({s['threshold_ipcw']:.2%})")
        lines.append(f"训练集Sensitivity: {s['sensitivity_ipcw_derivation']:.4f}")
        lines.append(f"训练集Specificity: {s['specificity_ipcw_derivation']:.4f}")
        lines.append("【模型验证：测试集】")
        lines.append(f"测试集总样本: {s['n_total_validation']}")
        lines.append(f"测试集10年内事件数: {s['n_events_before_10y_validation']}")
        lines.append(f"测试集随访超过10年对照数: {s['n_controls_after_10y_validation']}")
        lines.append(f"测试集10年前删失数: {s['n_censored_before_10y_validation']}")
        lines.append(f"测试集IPCW AUC: {s['auc_ipcw_validation']:.4f}")
        lines.append(f"训练集cutoff应用到测试集时的Sensitivity: {s['sensitivity_validation_at_derivation_cutoff']:.4f}")
        lines.append(f"训练集cutoff应用到测试集时的Specificity: {s['specificity_validation_at_derivation_cutoff']:.4f}")
        lines.append(f"训练集cutoff应用到测试集时的Youden: {s['youden_validation_at_derivation_cutoff']:.4f}")
        lines.append(f"测试集自身最佳cutoff（仅作对照，不用于网页）: {s['threshold_ipcw_validation_own_optimum']:.4f} ({s['threshold_ipcw_validation_own_optimum']:.2%})")
        lines.append(f"测试集IPCW Brier score: {s['brier_score_ipcw_validation']:.4f}")
        lines.append("-" * 40)
    text_path.write_text("\n".join(lines), encoding="utf-8")
    print_step("写出汇总文件")
    print(f"阈值CSV: {thresholds_path}")
    print(f"网页目录阈值CSV副本: {root_thresholds_path}")
    print(f"文字汇总: {text_path}")


def main() -> None:
    print("=" * 70)
    print("10年DCA、IPCW time-dependent ROC Youden和校准曲线生成脚本")
    print("=" * 70)
    ensure_output_dir()
    train, test = load_data()
    train = add_outcomes(train)
    test = add_outcomes(test)

    print_step("核对事件定义")
    print(f"训练集全因死亡: {train['all_cause_event'].sum()}/{len(train)}")
    print(f"测试集全因死亡: {test['all_cause_event'].sum()}/{len(test)}")
    print(f"训练集心血管死亡: {train['cardiovascular_event'].sum()}/{len(train)}")
    print(f"测试集心血管死亡: {test['cardiovascular_event'].sum()}/{len(test)}")
    print(f"测试集 ucod_leading 唯一值: {sorted(test['ucod_leading'].dropna().unique().tolist())}")

    summaries = []
    for task in TASKS:
        summaries.append(analyze_task(train, test, task))
    write_summary(summaries)

    print("\n全部完成。输出目录:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
