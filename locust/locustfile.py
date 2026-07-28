"""Load profile: mostly predictions, some status polls.

Run against the nginx front door so replica count is what varies:
    locust -f locust/locustfile.py --host http://localhost:8090
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_POOL: list[bytes] = []


def _load_pool(limit=40):
    """Read a few test images into memory once, so disk IO isn't measured."""
    pool = []
    for class_dir in sorted((PROJECT_ROOT / "data" / "test").iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("*.jpg"))[:limit // 4]:
            pool.append(path.read_bytes())
    return pool


class CoffeeBeanUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        global IMAGE_POOL
        if not IMAGE_POOL:
            IMAGE_POOL = _load_pool()

    @task(9)
    def predict(self):
        payload = random.choice(IMAGE_POOL)
        self.client.post(
            "/api/predict",
            files={"file": ("bean.jpg", payload, "image/jpeg")},
            name="/api/predict")

    @task(1)
    def status(self):
        self.client.get("/api/status", name="/api/status")
