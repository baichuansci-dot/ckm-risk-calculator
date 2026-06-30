import os

workers = 1
bind = "0.0.0.0:{}".format(os.environ.get("PORT", "5001"))
timeout = 300
preload_app = True  # Share models/SHAP across workers (saves memory on Render 512MB limit)
