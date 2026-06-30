# CKM Mortality Risk Calculator

Predicts 10-year all-cause and cardiovascular mortality risk for CKM Syndrome Stages 0–3. The 10-year risk is computed from survival models as 1 − S(120 months). High-risk thresholds are derived from the 10-year IPCW time-dependent ROC Youden index, and the calculator uses binary risk stratification: High Risk vs Low / Non-high Risk.

## Project Structure

```
deployment_package/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── railway.toml              # Railway deployment config
├── Procfile                  # Process file for deployment
├── nixpacks.toml             # Nixpacks build config
├── scaler.pkl                # Data scaler
├── 训练集_标准化后.csv        # Training data for SHAP background
├── models/
│   ├── CI_all_cause_death_GradientBoostingSurvival.pkl
│   └── CI_cardiovascular_death_RandomSurvivalForest.pkl
└── templates/
    └── index.html            # Web interface
```

## 🚀 Deploy to Railway

1. **Create GitHub Repository**:
   - Push this folder to a new GitHub repository

2. **Deploy on Railway**:
   - Go to [railway.app](https://railway.app) and sign in
   - Click **New Project** → **Deploy from GitHub repo**
   - Select your repository
   - Railway will auto-detect the configuration and deploy

3. **Generate Domain**:
   - Once deployed, click **Settings** → **Generate Domain**
   - Your app will be available at the generated URL

## ⚙️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will be available at `http://localhost:5001`

## 📝 Notes

- **Memory**: The app uses SHAP for model explanations which requires significant memory
- **Timeout**: Model predictions may take 10-30 seconds due to SHAP calculations
