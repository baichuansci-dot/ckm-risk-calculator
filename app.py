#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web-based Risk Calculator using Flask
Supports both All-cause and Cardiovascular Mortality Prediction
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

# Set font to Times New Roman globally for matplotlib
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # Use STIX font for math text which is similar to Times New Roman
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "训练集_标准化后.csv")

# Load models and scaler
print("Loading models and scaler...")
try:
    scaler = joblib.load(SCALER_PATH)
    print(f"✓ Scaler loaded successfully")
    model_all_cause = joblib.load(os.path.join(MODEL_DIR, "CI_all_cause_death_GradientBoostingSurvival.pkl"))
    print(f"✓ All-cause model loaded successfully")
    model_cardiovascular = joblib.load(os.path.join(MODEL_DIR, "CI_cardiovascular_death_RandomSurvivalForest.pkl"))
    print(f"✓ Cardiovascular model loaded successfully")
except Exception as e:
    print(f"✗ ERROR loading models: {e}")
    print(f"  Python version: {os.sys.version}")
    print(f"  joblib version: {joblib.__version__}")
    import sklearn
    print(f"  scikit-learn version: {sklearn.__version__}")
    raise

# Load training data for SHAP
df_train = pd.read_csv(DATA_PATH)

# Prediction Time Horizon (Months)
# Set to 120 months (10 years), consistent with external validation follow-up.
TIME_HORIZON = 120.0

# Get all feature names that scaler expects
if hasattr(scaler, 'feature_names_in_'):
    scaler_features = list(scaler.feature_names_in_)
else:
    scaler_features = list(df_train.columns)

# Rename HighCholesterol to Dyslipidemia for display purposes
if 'HighCholesterol' in df_train.columns:
    df_train = df_train.rename(columns={'HighCholesterol': 'Dyslipidemia'})

# Feature definitions
all_cause_features = [
    'WBC', 'Dyslipidemia', 'DBP', 'Creatinine', 'Glucose', 
    'Gender', 'TG', 'SBP', 'Age', 'MCV', 'smoking', 'Platelet', 'CI'
]

cardiovascular_features = [
    'WBC', 'DBP', 'UricAcid', 'SBP', 'BUN', 'Age', 'CI'
]

# Define categorical features (not standardized)
categorical_features = ['Gender', 'smoking', 'HighCholesterol', 'Dyslipidemia']

# Feature name mapping (English display names)
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

# Feature ranges and units
feature_info = {
    'WBC': {'min': 1.0, 'max': 30.0, 'step': 0.1, 'unit': '10^9/L', 'type': 'number'},
    'Dyslipidemia': {
        'min': 1, 'max': 2, 'step': 1, 'unit': '', 'type': 'select', 
        'options': [
            {'value': 1, 'label': 'Yes (have dyslipidemia)'}, 
            {'value': 2, 'label': 'No (no dyslipidemia)'}
        ]
    },
    'DBP': {'min': 40, 'max': 120, 'step': 1, 'unit': 'mmHg', 'type': 'number'},
    'Creatinine': {'min': 0.1, 'max': 15.0, 'step': 0.1, 'unit': 'mg/dL', 'type': 'number'},
    'Glucose': {'min': 50, 'max': 500, 'step': 1, 'unit': 'mg/dL', 'type': 'number'},
    'Gender': {
        'min': 1, 'max': 2, 'step': 1, 'unit': '', 'type': 'select', 
        'options': [
            {'value': 1, 'label': 'Male'}, 
            {'value': 2, 'label': 'Female'}
        ]
    },
    'TG': {'min': 20, 'max': 1000, 'step': 1, 'unit': 'mg/dL', 'type': 'number'},
    'SBP': {'min': 80, 'max': 220, 'step': 1, 'unit': 'mmHg', 'type': 'number'},
    'Age': {'min': 18, 'max': 100, 'step': 1, 'unit': 'years', 'type': 'number'},
    'MCV': {'min': 60, 'max': 120, 'step': 0.1, 'unit': 'fL', 'type': 'number'},
    'smoking': {
        'min': 0, 'max': 2, 'step': 1, 'unit': '', 'type': 'select', 
        'options': [
            {'value': 0, 'label': 'Never'}, 
            {'value': 1, 'label': 'Former'}, 
            {'value': 2, 'label': 'Current'}
        ]
    },
    'Platelet': {'min': 10, 'max': 1000, 'step': 1, 'unit': '10^9/L', 'type': 'number'},
    'CI': {'min': 1.0, 'max': 1.8, 'step': 0.01, 'unit': '', 'type': 'number'},
    'UricAcid': {'min': 1.0, 'max': 15.0, 'step': 0.1, 'unit': 'mg/dL', 'type': 'number'},
    'BUN': {'min': 1.0, 'max': 150, 'step': 0.1, 'unit': 'mg/dL', 'type': 'number'}
}

# Initialize SHAP explainers
print("Initializing SHAP explainers (this may take a moment)...")
# Use subset of training data for SHAP background to reduce memory footprint
# 2000 samples with 50 clusters balances explanation quality and memory (<512MB Render limit)
N_SHAP_SAMPLES = 2000
N_SHAP_CLUSTERS = 50
if len(df_train) > N_SHAP_SAMPLES:
    df_train_shap = df_train.sample(n=N_SHAP_SAMPLES, random_state=42)
else:
    df_train_shap = df_train
X_all_cause = df_train_shap[all_cause_features]
X_cardio = df_train_shap[cardiovascular_features]

print(f"Using {len(X_all_cause)} training samples for SHAP background data...")

def predict_all_cause(data):
    """Predict all-cause mortality probability at 10 years as 1 - S(120 months)."""
    if not hasattr(model_all_cause, "predict_survival_function"):
        raise RuntimeError("All-cause model must provide predict_survival_function for 10-year probability estimation.")

    surv_funcs = model_all_cause.predict_survival_function(data)
    death_probs = []
    for surv_func in surv_funcs:
        # 10-year mortality risk = 1 - survival probability at 120 months
        surv_prob = surv_func(TIME_HORIZON)
        death_prob = 1.0 - surv_prob
        death_probs.append(death_prob)
    return np.array(death_probs)

def predict_cardio(data):
    """Predict cardiovascular mortality probability at 10 years as 1 - S(120 months)."""
    if not hasattr(model_cardiovascular, "predict_survival_function"):
        raise RuntimeError("Cardiovascular model must provide predict_survival_function for 10-year probability estimation.")

    surv_funcs = model_cardiovascular.predict_survival_function(data)
    death_probs = []
    for surv_func in surv_funcs:
        # 10-year mortality risk = 1 - survival probability at 120 months
        surv_prob = surv_func(TIME_HORIZON)
        death_prob = 1.0 - surv_prob
        death_probs.append(death_prob)
    return np.array(death_probs)

# Calculate Risk Thresholds based on IPCW time-dependent ROC Youden Index (10-Year)
# Derived from risk_thresholds_10_year_Youden_IPCW.csv in this package.
print("Setting risk thresholds based on 10-Year IPCW time-dependent ROC Youden Index...")
THRESHOLD_PATH = os.path.join(BASE_DIR, "risk_thresholds_10_year_Youden_IPCW.csv")

def load_risk_thresholds():
    """Load high-risk thresholds derived from 10-year IPCW time-dependent ROC Youden index."""
    fallback_thresholds = {
        # Training-set derivation cutoffs. The test-set own optimum is reported
        # only as a validation contrast and should not drive web stratification.
        'all_cause': {'high': 0.07746522038517001},
        'cardio': {'high': 0.05471649432917636}
    }
    if not os.path.exists(THRESHOLD_PATH):
        print(f"Warning: {THRESHOLD_PATH} not found; using embedded IPCW Youden thresholds.")
        return fallback_thresholds

    threshold_df = pd.read_csv(THRESHOLD_PATH)
    loaded = {}
    for _, row in threshold_df.iterrows():
        if row['outcome'] == 'all_cause':
            loaded['all_cause'] = {'high': float(row['threshold_ipcw'])}
        elif row['outcome'] == 'cardiovascular':
            loaded['cardio'] = {'high': float(row['threshold_ipcw'])}

    for key, value in fallback_thresholds.items():
        loaded.setdefault(key, value)
    return loaded

thresholds = load_risk_thresholds()
print(f"Risk Thresholds (10-Year IPCW Youden):")
print(f"  All-cause: Low/Non-high < {thresholds['all_cause']['high']:.2%} | High >= {thresholds['all_cause']['high']:.2%}")
print(f"  Cardio:    Low/Non-high < {thresholds['cardio']['high']:.2%} | High >= {thresholds['cardio']['high']:.2%}")

# Create SHAP explainers
# Cluster training data into representative points for computational efficiency
# Using 50 cluster centers to stay within Render 512MB memory limit
print(f"Clustering training data for SHAP into {N_SHAP_CLUSTERS} centers...")
explainer_all_cause = shap.KernelExplainer(predict_all_cause, shap.kmeans(X_all_cause, N_SHAP_CLUSTERS))
explainer_cardio = shap.KernelExplainer(predict_cardio, shap.kmeans(X_cardio, N_SHAP_CLUSTERS))
print(f"SHAP explainers initialized: {len(X_all_cause)} training samples -> {N_SHAP_CLUSTERS} cluster centers")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         all_cause_features=all_cause_features,
                         cardiovascular_features=cardiovascular_features,
                         feature_mapping=feature_mapping,
                         feature_info=feature_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        data = request.json
        model_type = data.get('model_type', 'all_cause')
        
        # Select model and features
        if model_type == 'all_cause':
            features = all_cause_features
            explainer = explainer_all_cause
            predict_fn = predict_all_cause
        else:
            features = cardiovascular_features
            explainer = explainer_cardio
            predict_fn = predict_cardio
        
        # Extract input values
        input_values = []
        for feat in features:
            value = float(data.get(feat, 0))
            input_values.append(value)
        
        # Create DataFrame with original values
        input_df_original = pd.DataFrame([input_values], columns=features)
        
        # Map Dyslipidemia back to HighCholesterol for model compatibility
        input_df_for_model = input_df_original.copy()
        if 'Dyslipidemia' in input_df_for_model.columns:
            input_df_for_model = input_df_for_model.rename(columns={'Dyslipidemia': 'HighCholesterol'})
        
        # Separate categorical and continuous features
        model_features = list(input_df_for_model.columns)
        continuous_features_in_model = [f for f in model_features if f not in categorical_features]
        
        # Create full feature DataFrame for scaler
        input_full = pd.DataFrame(0.0, index=[0], columns=scaler_features)
        
        # Fill in continuous features
        for col in continuous_features_in_model:
            if col in scaler_features:
                input_full[col] = input_df_for_model[col].values[0]
        
        # Standardize continuous features
        continuous_standardized = scaler.transform(input_full)
        
        # Get standardized values
        continuous_standardized_dict = {}
        for col in continuous_features_in_model:
            if col in scaler_features:
                idx = scaler_features.index(col)
                continuous_standardized_dict[col] = continuous_standardized[0, idx]
        
        # Combine standardized continuous features with categorical features
        final_input = []
        for col in model_features:
            if col in categorical_features:
                final_input.append(input_df_for_model[col].values[0])
            else:
                final_input.append(continuous_standardized_dict[col])
        
        input_standardized = np.array([final_input])
        
        # Clip inputs to training data range to avoid extrapolation
        # This prevents the model from behaving unpredictably for out-of-distribution values
        try:
            # Map model features to training data features (handle Dyslipidemia/HighCholesterol mismatch)
            train_features = []
            for feat in model_features:
                if feat == 'HighCholesterol' and 'Dyslipidemia' in df_train.columns:
                    train_features.append('Dyslipidemia')
                elif feat == 'Dyslipidemia' and 'HighCholesterol' in df_train.columns:
                    train_features.append('HighCholesterol')
                else:
                    train_features.append(feat)

            # Get training data for the specific model features
            # Note: df_train has standardized values for continuous features and raw for categorical
            train_subset = df_train[train_features]
            
            # Calculate min and max from training data (using percentiles to avoid outliers)
            # Using 0.5th and 99.5th percentiles to be robust against outliers
            train_min = train_subset.quantile(0.005).values
            train_max = train_subset.quantile(0.995).values
            
            # Clip the input
            input_standardized = np.clip(input_standardized, train_min, train_max)
            print(f"Input clipped to training range (0.5-99.5 percentile) for stability.")
        except Exception as e:
            print(f"Warning: Could not clip input to training range: {e}")

        # Make prediction
        prediction = float(predict_fn(input_standardized)[0])
        
        # Determine Risk Level (binary stratification)
        model_key = 'all_cause' if model_type == 'all_cause' else 'cardio'
        high_thresh = thresholds[model_key]['high']

        if prediction >= high_thresh:
            risk_level = "High Risk"
            risk_color = "red"
            risk_desc = f"At or above the 10-year IPCW Youden cutoff ({high_thresh:.2%})"
        else:
            risk_level = "Low / Non-high Risk"
            risk_color = "green"
            risk_desc = f"Below the 10-year IPCW Youden cutoff ({high_thresh:.2%})"

        # Generate prediction label
        if model_type == 'all_cause':
            prediction_label = f"10-Year All-Cause Mortality Risk: {prediction:.2%}"
            prediction_note = (f"Risk Level: {risk_level}. "
                               f"High Risk is defined as predicted 10-year risk ≥ {high_thresh:.2%}, "
                               "derived from the 10-year IPCW time-dependent ROC Youden index. "
                               "Risk is calculated as 1 − S(120 months).")
        else:
            prediction_label = f"10-Year Cardiovascular Mortality Risk: {prediction:.2%}"
            prediction_note = (f"Risk Level: {risk_level}. "
                               f"High Risk is defined as predicted 10-year risk ≥ {high_thresh:.2%}, "
                               "derived from the 10-year IPCW time-dependent ROC Youden index. "
                               "Risk is calculated as 1 − S(120 months).")
        
        # Calculate SHAP values (nsamples=500 balances precision with Render 512MB limit)
        shap_values = explainer.shap_values(input_standardized, nsamples=500)
        
        # Handle list output from explainer
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        # Get base value
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            if len(base_value) > 1:
                base_value = base_value[1] if len(base_value) == 2 else base_value[0]
            else:
                base_value = base_value[0]
        base_value = float(base_value)
        
        # Create display names
        display_names = [feature_mapping.get(f, f) for f in features]
        
        # Create SHAP explanation object
        shap_explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=input_df_original.values[0],
            feature_names=display_names
        )
        
        # Generate waterfall plot
        shap.plots.waterfall(shap_explanation, show=False, max_display=15)
        
        # Convert to base64 image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # Prepare SHAP contributions for display
        shap_contributions = []
        for i, feat in enumerate(features):
            shap_contributions.append({
                'feature': feature_mapping.get(feat, feat),
                'value': float(input_df_original.iloc[0, i]),
                'shap_value': float(shap_values[0][i])
            })
        
        # Sort by absolute SHAP value
        shap_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
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
            'base_value': base_value,
            'shap_plot': image_base64,
            'shap_contributions': shap_contributions
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Starting Web Calculator Application")
    print("="*70)
    print("\nFeature Encoding:")
    print("  - Dyslipidemia: 1 = Yes (have dyslipidemia), 2 = No (no dyslipidemia)")
    print("  - Sex (Gender): 1 = Male, 2 = Female")
    print("  - Smoking: 0 = Never, 1 = Former, 2 = Current")
    print("\nServer starting on http://0.0.0.0:5001")
    print("="*70 + "\n")
    port = int(os.environ.get('PORT', 5001))
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
