#!/usr/bin/env bash
# Runs an identical load profile at 1, 2, and 4 API replicas.
#
# This host is 4 cores / 7GB. Stop other containers first or the 4-replica
# run will be memory starved:
#     docker stop $(docker ps -q --filter name=supabase)
set -euo pipefail

USERS=50
SPAWN_RATE=5
DURATION=2m
HOST=http://localhost:8090
LOCUST=${LOCUST:-.venv/bin/locust}

mkdir -p locust/results

for N in 1 2 4; do
    echo "=== Scaling API to $N replica(s) ==="
    docker-compose up -d --scale api="$N"
    echo "Waiting for containers to warm up..."
    sleep 60

    until curl -sf "$HOST/api/health" > /dev/null; do sleep 2; done

    echo "=== Running load test with $N replica(s) ==="
    "$LOCUST" -f locust/locustfile.py \
        --host "$HOST" \
        --users "$USERS" \
        --spawn-rate "$SPAWN_RATE" \
        --run-time "$DURATION" \
        --headless \
        --csv "locust/results/replicas_${N}" \
        --csv-full-history

    echo "=== Done: $N replica(s) ==="
    sleep 10
done

echo
echo "Summary:"
for N in 1 2 4; do
    echo "--- $N replica(s) ---"
    column -s, -t < "locust/results/replicas_${N}_stats.csv" | head -3
done
