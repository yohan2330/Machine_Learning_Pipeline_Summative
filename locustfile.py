"""Locust load test: simulates many users sending images to the API.

Interface in the browser:
    locust -f locustfile.py --host http://localhost:8080
    then open http://localhost:8089

Without interface (saves the results in results/1_container_*.csv):
    locust -f locustfile.py --host http://localhost:8080 --headless \
           --users 50 --spawn-rate 10 --run-time 60s --csv results/1_container
"""
import os
import random

from locust import HttpUser, between, task

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test")

# Load some test images in memory once (so the test measures the API, not the disk)
IMAGES = []
for cls in ("Parasitized", "Uninfected"):
    folder = os.path.join(TEST_DIR, cls)
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder))[:20]:
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                with open(os.path.join(folder, name), "rb") as f:
                    IMAGES.append((name, f.read()))

if not IMAGES:
    raise SystemExit("No test images found in data/test/. Add the sample images first.")


class MalariaUser(HttpUser):
    wait_time = between(0.1, 0.5)     # a user waits 0.1 to 0.5 s between two requests

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        self.client.post("/predict", files={"file": (name, data, "image/png")}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
