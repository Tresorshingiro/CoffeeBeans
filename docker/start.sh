#!/usr/bin/env bash
set -e

# Optional: pull the full dataset if HF_DATA_REPO is configured. Falls back
# to the committed sample silently.
python scripts/download_data.py || echo "dataset download skipped"
python scripts/build_insights.py || echo "insights generation skipped"

# Without this the model registry is empty, the promotion gate compares
# against 0.0, and any retrained model would be promoted regardless of
# quality.
python scripts/register_baseline.py || echo "baseline registration skipped"

uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1 &
API_PID=$!

streamlit run ui/app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --browser.gatherUsageStats false &
UI_PID=$!

nginx -c /app/docker/nginx.space.conf -g 'daemon off;' &
NGINX_PID=$!

# If any process dies, take the container down so the Space restarts it.
wait -n $API_PID $UI_PID $NGINX_PID
exit $?
