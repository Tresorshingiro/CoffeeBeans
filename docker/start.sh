#!/usr/bin/env bash
set -e

# Nothing is computed here on purpose — every second spent before nginx binds
# :7860 is a second the Space shows "Starting" with no UI.
#
#   - data/insights.json is committed (built from the full 8,000-image set, so
#     it cannot be regenerated from the committed sample without contradicting
#     its own interpretation text).
#   - The champion is registered at image build time; see Dockerfile.space.
#   - scripts/download_data.py is deliberately not run: the baked baseline was
#     scored against data/test, and pulling the full dataset at boot would swing
#     config.test_dir() over to data/full/test, leaving the gate comparing the
#     champion and its candidates on two different eval slices.

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
