# Malaria Detection: End-to-End ML Pipeline

Classifies a blood cell image as **Parasitized** (infected) or **Uninfected** (healthy).
The project covers data processing, model training, prediction, retraining with a trigger, an API, a web interface, deployment and a load test.

## Links

- **Video demo (YouTube):** _paste the link here_
- **Deployed app:** https://machine-learning-pipeline-summative-1.onrender.com
- **API docs:** https://machine-learning-pipeline-summative-1.onrender.com/docs
- **GitHub:** https://github.com/yohan2330/Machine_Learning_Pipeline_Summative

> The free Render plan sleeps after 15 minutes without traffic (the first request can take about one minute) and has limited memory.
> Images uploaded and models retrained online are lost when the service restarts. The full demo is in the video.

## Description

- **Data:** [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) (Kaggle), images of single blood cells, 2 classes. A small sample is in `data/`.
- **Model:** MobileNetV2 (transfer learning), dropout, Adam optimizer, images resized to 128x128.
- **Retraining:** the user uploads new images of one class, the API reports `retrain_needed` when 50 images are waiting (the trigger), and the **Retrain** button trains the saved model again with the old and new images.

## Results (test set, notebook)

| Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| 0.901 | 0.874 | 0.937 | 0.904 | 0.967 |

## Project structure

```
├── README.md
├── notebook/malaria_detection.ipynb   # preprocessing, training, testing
├── src/
│   ├── preprocessing.py
│   ├── model.py                       # build, train, evaluate, retrain
│   ├── prediction.py
│   └── api.py                         # FastAPI + web interface
├── static/                            # web interface (HTML, CSS, JS, charts)
├── data/train, data/test              # sample images
├── models/malaria_model.h5
├── Dockerfile, docker-compose.yml, nginx.conf
├── locustfile.py                      # load test
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/yohan2330/Machine_Learning_Pipeline_Summative.git
cd Machine_Learning_Pipeline_Summative
pip install -r requirements.txt
python -m uvicorn src.api:app --port 8000
```

Open http://localhost:8000 (interface) or http://localhost:8000/docs (API docs).

**With Docker:**

```bash
docker build -t malaria-api .
docker run -p 8000:8000 malaria-api
```

## Web interface

| Tab | What it does |
|---|---|
| Dashboard | Model up-time, evaluation on the test set, last training info |
| Predict | Upload one image, get the class and the confidence |
| Data visualizations | Class balance and 3 features (brightness, contrast, image size) with interpretations |
| Upload & Retrain | Upload many images, see the retraining trigger, retrain the model |

## API

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Status and up-time |
| GET | `/status` | Model info, pending uploads, `retrain_needed` |
| POST | `/predict` | Predict one image |
| POST | `/upload` | Upload many images of one class |
| POST | `/retrain` | Retrain with the uploaded images (background) |
| POST | `/evaluate` | Evaluate the current model on `data/test` (background) |
| GET | `/evaluation-status` | Result of the evaluation |

## Load test (Locust)

50 users, 60 seconds, requests sent to `POST /predict` through an nginx load balancer.

```bash
docker compose up --build -d --scale api=3      # 1, 2 or 3 containers
python -m locust -f locustfile.py --host http://localhost:8080 --headless --users 50 --spawn-rate 10 --run-time 60s
```

| Containers | Requests | Failures | Requests/s | Average (ms) | Median (ms) | 95th percentile (ms) | Max (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 440 | 0 | 7.4 | 5402 | 5400 | 6300 | 6554 |
| 2 | 705 | 0 | 11.8 | 3314 | 2100 | 7100 | 7497 |
| 3 | 864 | 0 | 14.5 | 2576 | 2500 | 4900 | 5757 |

With 3 containers the API handles about twice more requests per second, and the average response time drops from 5.4 s to 2.6 s.
The gain is not 3 times because all containers share the same computer (same CPU) and TensorFlow needs a lot of CPU for each prediction.