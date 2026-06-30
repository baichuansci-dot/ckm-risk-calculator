#!/bin/bash
# Mortality Prediction Web Calculator - Startup Script
# Auto-starts gunicorn server and serveo tunnel

DEPLOY_DIR="/Users/gubaichuan/Desktop/心肾综合症1019/0-3期/重新算1110/重重新1115/1118重新0-3期-机器学习/我的文章！！！/终稿/0622根据EJPC建议修改/deployment_package_10year_final"
LOG_DIR="$DEPLOY_DIR/logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Starting Mortality Prediction Web Calculator"
echo "============================================"

# Activate venv and start gunicorn
cd "$DEPLOY_DIR"
source venv/bin/activate

# Check if gunicorn already running
if pgrep -f "gunicorn.*app:app" > /dev/null; then
    echo "[$(date)] Gunicorn already running, skipping..."
else
    echo "[$(date)] Starting Gunicorn..."
    PORT=5001 gunicorn -c gunicorn_config.py app:app \
        --access-logfile "$LOG_DIR/gunicorn_access.log" \
        --error-logfile "$LOG_DIR/gunicorn_error.log" \
        >> "$LOG_DIR/gunicorn.log" 2>&1 &
    echo "[$(date)] Gunicorn started with PID $!"
fi

# Start serveo tunnel (with auto-reconnect loop)
if pgrep -f "serveo.net" > /dev/null; then
    echo "[$(date)] Serveo tunnel already running, skipping..."
else
    echo "[$(date)] Starting Serveo tunnel..."
    while true; do
        echo "[$(date)] Connecting to serveo.net..."
        ssh -o StrictHostKeyChecking=no \
            -o ConnectTimeout=10 \
            -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=3 \
            -o TCPKeepAlive=yes \
            -R 80:localhost:5001 serveo.net \
            2>&1 | tee -a "$LOG_DIR/serveo.log"
        echo "[$(date)] Serveo disconnected, reconnecting in 10s..."
        sleep 10
    done &
    echo "[$(date)] Serveo tunnel started with PID $!"
fi

echo ""
echo "Server should be accessible at:"
echo "  Local:  http://127.0.0.1:5001"
echo "  Public: (check serveo log for URL)"
echo "          grep 'Forwarding' $LOG_DIR/serveo.log | tail -1"
echo ""
