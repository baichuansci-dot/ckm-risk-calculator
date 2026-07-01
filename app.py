#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web-based Risk Calculator using Flask
10-Year All-cause and Cardiovascular Mortality Prediction in CKM Syndrome Stages 0-3
Memory-optimized for Render 512MB free tier
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import shap
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['figure.max_open_warning'] = 0

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "训练集_标准化后.csv")

# ------------------------------------------------------------
# 1. Load only essentials at startup (scaler + models ONLY)
# ------------------------------------------------------------
print("=" * 50)
print("Starting CKM Risk Calculator (memory-optimized)")
print("=" * 50)

print("[1/4] Loading scaler...")
scaler = joblib.load(SCALER_PATH)
print(f"  ✓ Scaler loaded")

print("[2/4] Loading all-cause model...")
model_all_cause = joblib.load(os.path.join(MODEL_DIR, "CI_all_cause_death_GradientBoostingSurvival.pkl"))
print(f"  ✓ All-cause model loaded")

print("[3/4] Loading cardiovascular model...")
model_cardiovascular = joblib.load(os.path.join(MODEL_DIR, "CI_cardiovascular_death_RandomSurvivalForest.pkl"))
print(f"  ✓ Cardiovascular model loaded")

# ------------------------------------------------------------
# 2. Load minimal training data (for input clipping only)
# ------------------------------------------------------------
print("[4/4] Loading reference data...")
needed_cols = [
    'WBC', 'Dyslipidemia', 'HighCholesterol', 'DBP', 'Creatinine', 'Glucose',
    'Gender', 'TG', 'SBP', 'Age', 'MCV', 'smoking', 'Platelet', 'CI',
    'UricAcid', 'BUN'
]
available_cols = [c for c in needed_cols if c in pd.read_csv(DATA_PATH, nrows=1).columns]
df_train_full = pd.read_csv(DATA_PATH, usecols=available_cols)
if 'HighCholesterol' in df_train_full.columns:
    df_train_full = df_train_full.rename(columns={'HighCholesterol': 'Dyslipidemia'})

# Pre-compute clipping bounds (0.5-99.5 percentile) — stored as small arrays
clip_bounds = {}
for feat_set_name, feat_list in [
    ('all_cause', ['WBC', 'Dyslipidemia', 'DBP', 'Creatinine', 'Glucose',
                   'Gender', 'TG', 'SBP', 'Age', 'MCV', 'smoking', 'Platelet', 'CI']),
    ('cardio', ['WBC', 'DBP', 'UricAcid', 'SBP', 'BUN', 'Age', 'CI'])
]:
    cols = [c for c in feat_list if c in df_train_full.columns]
    sub = df_train_full[cols]
    clip_bounds[feat_set_name] = {
        'min': sub.quantile(0.005).values,
        'max': sub.quantile(0.995).values,
        'cols': cols
    }
print(f"  ✓ Clipping bounds computed for {len(available_cols)} features")

# ------------------------------------------------------------
# 3. Constants
# ------------------------------------------------------------
TIME_HORIZON = 120.0

if hasattr(scaler, 'feature_names_in_'):
    scaler_features = list(scaler.feature_names_in_)
else:
    scaler_features = list(df_train_full.columns)

all_cause_features = [
    'WBC', 'Dyslipidemia', 'DBP', 'Creatinine', 'Glucose',
    'Gender', 'TG', 'SBP', 'Age', 'MCV', 'smoking', 'Platelet', 'CI'
]
cardiovascular_features = [
    'WBC', 'DBP', 'UricAcid', 'SBP', 'BUN', 'Age', 'CI'
]
categorical_features = ['Gender', 'smoking', 'HighCholesterol', 'Dyslipidemia']

feature_mapping = {
    'WBC': 'White Blood Cell (10^9/L)',
    'Dyslipidemia': 'Dyslipidemia',
    'DBP': 'Diastolic Blood Pressure (mmHg)',
    'Creatinine': 'Serum Creatinine (mg/dL)',
    'Glucose': 'Blood Glucose (mg/dL)',
    'Gender': 'Sex',
    'TG': 'Triglycerides (mg/dL)',
    'SBP': 'Systolic Blood Pressure (mmHg)',
    'Age': 'Age (years)',
    'MCV': 'Mean Corpuscular Volume (fL)',
    'smoking': 'Smoking Status',
    'Platelet': 'Platelet Count (10^9/L)',
    'CI': 'Conicity Index',
    'UricAcid': 'Uric Acid (mg/dL)',
    'BUN': 'Blood Urea Nitrogen (mg/dL)'
}

feature_info = {
    'WBC': {'min': 1.0, 'max': 30.0, 'step': 0.1, 'unit': '10^9/L', 'type': 'number'},
    'Dyslipidemia': {
        'min': 1, 'max': 2, 'step': 1, 'unit': '', 'type': 'select',
        'options': [{'value': 1, 'label': 'Yes (have dyslipidemia)'}, {'value': 2, 'label': 'No (no dyslipidemia)'}]
    },
    'DBP': {'min': 40, 'max': 120, 'step': 1, 'unit': 'mmHg', 'type': 'number'},
    'Creatinine': {'min': 0.1, 'max': 15.0, 'step': 0.1, 'unit': 'mg/dL', 'type': 'number'},
    'Glucose': {'min': 50, 'max': 500, 'step': 1, 'unit': 'mg/dL', 'type': 'number'},
    'Gender': {
        'min': 1, 'max': 2, 'step': 1, 'unit': '', 'type': 'select',
        'options': [{'value': 1, 'label': 'Male'}, {'value': 2, 'label': 'Female'}]
    },
    'TG': {'min': 20, 'max': 1000, 'step': 1, 'unit': 'mg/dL', 'type': 'number'},
    'SBP': {'min': 80, 'max': 220, 'step': 1, 'unit': 'mmHg', 'type': 'number'},
    'Age': {'min': 18, 'max': 100, 'step': 1, 'unit': 'years', 'type': 'number'},
    'MCV': {'min': 60, 'max': 120, 'step': 0.1, 'unit': 'fL', 'type': 'number'},
    'smoking': {
        'min': 0, 'max': 2, 'step': 1, 'unit': '', 'type': 'select',
        'options': [{'value': 0, 'label': 'Never'}, {'value': 1, 'label': 'Former'}, {'value': 2, 'label': 'Current'}]
    },
    'Platelet': {'min': 10, 'max': 1000, 'step': 1, 'unit': '10^9/L', 'type': 'number'},
    'CI': {'min': 1.0, 'max': 1.8, 'step': 0.01, 'unit': '', 'type': 'number'},
    'UricAcid': {'min': 1.0, 'max': 15.0, 'step': 0.1, 'unit': 'mg/dL', 'type': 'number'},
    'BUN': {'min': 1.0, 'max': 150, 'step': 0.1, 'unit': 'mg/dL', 'type': 'number'}
}

# ------------------------------------------------------------
# 4. Prediction functions
# ------------------------------------------------------------
def predict_all_cause(data):
    surv_funcs = model_all_cause.predict_survival_function(data)
    return np.array([1.0 - sf(TIME_HORIZON) for sf in surv_funcs])

def predict_cardio(data):
    surv_funcs = model_cardiovascular.predict_survival_function(data)
    return np.array([1.0 - sf(TIME_HORIZON) for sf in surv_funcs])

# ------------------------------------------------------------
# 5. Risk thresholds
# ------------------------------------------------------------
THRESHOLD_PATH = os.path.join(BASE_DIR, "risk_thresholds_10_year_Youden_IPCW.csv")
_fallback = {
    'all_cause': {'high': 0.07746522038517001},
    'cardio': {'high': 0.05471649432917636}
}
thresholds = dict(_fallback)
if os.path.exists(THRESHOLD_PATH):
    tdf = pd.read_csv(THRESHOLD_PATH)
    for _, row in tdf.iterrows():
        thresholds[row['outcome']] = {'high': float(row['threshold_ipcw'])}
print(f"  ✓ Thresholds: All-cause≥{thresholds['all_cause']['high']:.1%}, Cardio≥{thresholds['cardio']['high']:.1%}")

# ------------------------------------------------------------
# 6. SHAP explainers — created LAZILY on first prediction request
#    (not at startup! avoids OOM during deployment)
# ------------------------------------------------------------
_explainers = {}       # cached explainers: {'all_cause': ..., 'cardio': ...}
_explainer_lock = {}   # prevents race conditions

def _get_explainer(model_type):
    """Create (or return cached) SHAP explainer for the given model type.

    Created lazily on first prediction to keep startup memory minimal.
    TreeExplainer is used for the GradientBoosting model (no background
    data needed, very memory efficient). KernelExplainer with minimal
    settings (50 background, 10 clusters) is used for RandomSurvivalForest.
    """
    if model_type in _explainers:
        return _explainers[model_type]

    print(f"[SHAP] Creating explainer for {model_type} (first request)...")

    # Tiny background sample for KernelExplainer (if needed)
    bg_n = 50
    bg_df = df_train_full.sample(n=min(bg_n, len(df_train_full)), random_state=42)

    if model_type == 'all_cause':
        try:
            expl = shap.TreeExplainer(model_all_cause, feature_perturbation="interventional")
            print(f"  ✓ All-cause TreeExplainer ready")
        except Exception:
            print(f"  TreeExplainer failed, using KernelExplainer...")
            X = bg_df[[c for c in all_cause_features if c in bg_df.columns]]
            expl = shap.KernelExplainer(predict_all_cause, shap.kmeans(X, 10))
            print(f"  ✓ All-cause KernelExplainer ready")
    else:
        X = bg_df[[c for c in cardiovascular_features if c in bg_df.columns]]
        expl = shap.KernelExplainer(predict_cardio, shap.kmeans(X, 10))
        print(f"  ✓ Cardio KernelExplainer ready")

    _explainers[model_type] = expl
    return expl

# ------------------------------------------------------------
# 7. Routes
# ------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html',
                           all_cause_features=all_cause_features,
                           cardiovascular_features=cardiovascular_features,
                           feature_mapping=feature_mapping,
                           feature_info=feature_info)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_type = data.get('model_type', 'all_cause')

        if model_type == 'all_cause':
            features = all_cause_features
            predict_fn = predict_all_cause
            feat_set_key = 'all_cause'
        else:
            features = cardiovascular_features
            predict_fn = predict_cardio
            feat_set_key = 'cardio'

        explainer = _get_explainer(model_type)

        # Build input DataFrame
        input_values = [float(data.get(f, 0)) for f in features]
        input_df_original = pd.DataFrame([input_values], columns=features)

        input_df_for_model = input_df_original.copy()
        if 'Dyslipidemia' in input_df_for_model.columns:
            input_df_for_model = input_df_for_model.rename(columns={'Dyslipidemia': 'HighCholesterol'})

        model_features = list(input_df_for_model.columns)
        continuous_feats = [f for f in model_features if f not in categorical_features]

        # Standardize
        input_full = pd.DataFrame(0.0, index=[0], columns=scaler_features)
        for col in continuous_feats:
            if col in scaler_features:
                input_full[col] = input_df_for_model[col].values[0]
        cont_std = scaler.transform(input_full)

        cont_std_dict = {}
        for col in continuous_feats:
            if col in scaler_features:
                idx = scaler_features.index(col)
                cont_std_dict[col] = cont_std[0, idx]

        final_input = []
        for col in model_features:
            if col in categorical_features:
                final_input.append(input_df_for_model[col].values[0])
            else:
                final_input.append(cont_std_dict[col])
        input_std = np.array([final_input])

        # Clip to training range
        bounds = clip_bounds.get(feat_set_key)
        if bounds:
            try:
                col_indices = [model_features.index(c) if c in model_features
                               else (model_features.index('Dyslipidemia') if c == 'HighCholesterol' and 'Dyslipidemia' in model_features
                               else None)
                               for c in bounds['cols']]
                clip_min = np.zeros(len(model_features))
                clip_max = np.ones(len(model_features)) * 999
                for i, ci in enumerate(col_indices):
                    if ci is not None:
                        clip_min[ci] = bounds['min'][i]
                        clip_max[ci] = bounds['max'][i]
                input_std = np.clip(input_std, clip_min, clip_max)
            except Exception:
                pass

        # Predict
        prediction = float(predict_fn(input_std)[0])

        # Risk level
        high_thresh = thresholds[feat_set_key]['high']
        if prediction >= high_thresh:
            risk_level = "High Risk"
            risk_color = "red"
            risk_desc = f"At or above the 10-year IPCW Youden cutoff ({high_thresh:.2%})"
        else:
            risk_level = "Low / Non-high Risk"
            risk_color = "green"
            risk_desc = f"Below the 10-year IPCW Youden cutoff ({high_thresh:.2%})"

        label_prefix = "10-Year All-Cause" if model_type == 'all_cause' else "10-Year Cardiovascular"
        prediction_label = f"{label_prefix} Mortality Risk: {prediction:.2%}"
        prediction_note = (
            f"Risk Level: {risk_level}. "
            f"High Risk ≥ {high_thresh:.2%} (10-year IPCW time-dependent ROC Youden index). "
            f"Risk = 1 − S(120 months)."
        )

        # SHAP
        shap_raw = explainer.shap_values(input_std, nsamples=150, silent=True)
        if isinstance(shap_raw, list):
            shap_raw = shap_raw[1] if len(shap_raw) == 2 else shap_raw[0]

        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]
        base_val = float(base_val)

        display_names = [feature_mapping.get(f, f) for f in features]

        shap_exp = shap.Explanation(
            values=shap_raw[0],
            base_values=base_val,
            data=input_df_original.values[0],
            feature_names=display_names
        )

        # Waterfall plot (low-res to save memory)
        shap.plots.waterfall(shap_exp, show=False, max_display=10)
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close('all')
        buf.close()

        # SHAP contributions
        shap_contribs = sorted(
            [{'feature': display_names[i],
              'value': float(input_df_original.iloc[0, i]),
              'shap_value': float(shap_raw[0][i])}
             for i in range(len(features))],
            key=lambda x: abs(x['shap_value']), reverse=True
        )

        return jsonify({
            'success': True,
            'prediction': prediction,
            'prediction_label': prediction_label,
            'prediction_note': prediction_note,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_desc': risk_desc,
            'threshold_used': high_thresh,
            'threshold_method': '10-year IPCW time-dependent ROC Youden index',
            'base_value': base_val,
            'shap_plot': img_b64,
            'shap_contributions': shap_contribs
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Feature Encoding:")
    print("  Dyslipidemia: 1=Yes, 2=No | Sex: 1=Male, 2=Female")
    print("  Smoking: 0=Never, 1=Former, 2=Current")
    print("Server: http://0.0.0.0:" + str(os.environ.get('PORT', 5001)))
    print("=" * 70 + "\n")
    port = int(os.environ.get('PORT', 5001))
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
