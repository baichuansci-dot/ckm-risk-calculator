import os

workers = 4
bind = "0.0.0.0:{}".format(os.environ.get("PORT", "5001"))
timeout = 300
