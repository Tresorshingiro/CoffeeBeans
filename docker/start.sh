#!/usr/bin/env bash
set -e

uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1 &
API_PID=$!

streamlit run ui/app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --server.fileWatcherType none \
    --server.enableXsrfProtection false \
    --server.enableCORS false \
    --browser.gatherUsageStats false &
UI_PID=$!

nginx -c /app/docker/nginx.space.conf -g 'daemon off;' &
NGINX_PID=$!

# If any process dies, take the container down so the Space restarts it.
wait -n $API_PID $UI_PID $NGINX_PID
exit $?
