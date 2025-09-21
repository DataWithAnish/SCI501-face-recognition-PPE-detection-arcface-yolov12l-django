# PPE Compliance with Identity (YOLOv12l + ArcFace)

Real time PPE monitoring with identity. The app detects PPE, identifies enrolled workers, overlays results on live video, and writes audit ready logs. It runs as a Django web app and is designed for edge use.

---

## Contents
- Overview
- System architecture
- Features
- Repo layout
- Requirements and install
- Models
- Quick start
- Endpoints
- Configuration
- Logs and outputs
- Evaluation and performance
- Training notes
- Privacy and ethics
- Troubleshooting
- Data and code availability
- Citation and acknowledgements
- License

---

## Overview

This project joins a YOLOv12l PPE detector with ArcFace identity. Faces are detected on a cadence, tracked in between, and re embedded sparsely. A single box per person is drawn with the name or **Unknown** and any PPE tags. The app streams MJPEG to a browser and saves CSV and JSON logs for audits.

---

## System architecture



![System architecture](https://github.com/DataWithAnish/SCI501-face-recognition-PPE-detection-arcface-yolov12l-django/blob/ed4c5d00d58334e860e7dd8104211ace4f50b457/%20ArcFace%20%2B%20YOLO%20Face%20Recognition%20%26%20PPE%20Overlay%20System%20Flow.png)

---

## Features

- PPE detection for hardhat, vest, goggles, mask, gloves and the matching no PPE tags
- ArcFace identity with mean embedding enrollment
- One box overlay per person with attached PPE tags
- Live MJPEG stream and a simple HTML page
- Session logs for identity, PPE, timing, summary, and debug
- Works with webcam, file, or RTSP IP camera

---

## Repo layout

recognition/
arcface_yolo.py # main pipeline
sessions/<session_id>/ # per run artifacts
workers_db.npy # mean embeddings per worker (auto built)
worker_app/
views.py # Django views
templates/workers/.html
media/workers/<name>/.jpg # enrollment photos per worker
requirements.txt
manage.py

---

## Requirements and install

- Python 3.9 or newer
- Optional GPU with CUDA for higher FPS

git clone <your-repo-url>
cd <your-repo>
python -m venv .venv
source .venv/bin/activate   (Windows: .venv\\Scripts\\activate)
pip install --upgrade pip
pip install -r requirements.txt

---

## Models

PPE detector uses a trained YOLOv12l checkpoint best.pt

Face analysis uses the InsightFace pack buffalo_l

detector file det_10g.onnx

ArcFace embeddings

Point the app to your YOLO model path. Use an absolute path, for example:
/path/to/models/best.pt
Configure this where yolo_model_path is read in your code.

---

## Quick start

Enroll workers (web path)
1. Open /workers
2. Go to /workers/upload
3. Enter a worker name and upload several clear face photos
The app builds or updates workers_db.npy with a mean L2 normalized embedding per worker.

Enroll workers (file path)
1. Create media/workers/<name>/
2. Place .jpg files inside
3. Visit /workers/upload once to trigger a rebuild

Run the server
python manage.py runserver

Watch the stream
Open http://127.0.0.1:8000/video

Or use a direct stream URL
Webcam → http://127.0.0.1:8000/video_feed/?source=camera
Video file → http://127.0.0.1:8000/video_feed/?source=/full/path/to/file.mp4
RTSP camera → http://127.0.0.1:8000/video_feed/?source=rtsp://user:pass@ip:554/stream

Optional identity threshold
Append &threshold=0.35 to the URL

---

## Endpoints

/ → redirects to /video
/video → live stream page
/video_feed → MJPEG stream (source and threshold query params)
/video_stats → JSON snapshot with counts, FPS, and session paths
/workers → list enrolled workers
/workers/upload → upload images and rebuild workers_db.npy

---

## Configuration

Core knobs in the pipeline
- DETECT_EVERY_N frames for face detect and hard reset of trackers
- EMBED_EVERY_M frames for recomputing embeddings
- YOLO_EVERY_N frames for PPE detection
- threshold identity cutoff (cosine similarity on L2 normalized vectors)
- PPE_OVERLAY_IOU to attach a PPE detection to a face track

You can pass threshold in the query string. Cadence values can be constants or loaded from a small config.

---

## Logs and outputs

Each run creates a folder under recognition/sessions/<session_id>/
- predictions_<id>.csv — identity scores and decisions
- yolo_<id>.csv — PPE detections per frame
- perf_<id>.csv — timings and FPS
- summary_<id>.json — session summary
- debug.log — concise prints for troubleshooting

These files are ready for dashboards and audits.

---

## Evaluation and performance

Detector at 640x640
- mAP50 0.780
- mAP50–95 0.525
- precision 0.717
- recall 0.830

End to end throughput in the web app
- Duty cycled demo on CPU about 12–13 FPS
- Sequential run where ArcFace then YOLO about 7–8 FPS
- Mean per processed frame times (from perf.csv)
  - ArcFace about 77 ms
  - YOLO about 52 ms
  - Total about 130 ms

Strong classes include ladder, person, goggles, and hardhat. Mask and No Safety Vest are weaker. Consider inferring no vest by absence inside the person region rather than learning it as a separate class.

---

## Training notes

If you want to train or fine tune with Ultralytics, typical arguments
yolo detect train model=yolov12l.pt data=path/to/data.yaml imgsz=640 epochs=100 batch=16 optimizer=SGD lr0=0.01 momentum=0.937 weight_decay=0.0005 device=0

Validate
yolo detect val model=path/to/best.pt data=path/to/data.yaml imgsz=640

---

## Privacy and ethics

Use enrollment photos with consent. The system stores embeddings and logs, not raw face crops from the stream. Low confidence cases are labeled Unknown. Access to logs is limited to authorized users. Define retention and deletion in policy.

---

## Troubleshooting

Two boxes for one person → lower DETECT_EVERY_N and ensure the one box owner policy is active.
Low FPS → increase cadence values for detect and PPE; enable GPU for ArcFace if available.
Many Unknown for an enrolled worker → add a few better photos and rebuild workers_db.npy; tune threshold slightly lower with care.

---

## Data

Dataset
Roboflow Universe — Personal Protective Equipment Combined Model
https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model
License CC BY 4.0



---

## Citation and acknowledgements

If you use this work please cite
- ArcFace — Deng J, Guo J, et al. 2019
- InsightFace model zoo buffalo_l (det_10g and ArcFace)
- Ultralytics YOLOv12
- Roboflow Universe PPE Combined dataset

Acknowledgements
University of New England for support and computing. Thanks to supervisor Fareed Ud Din and unit coordinator Michelle Taylor for guidance and feedback.

---

## License

Add your code license here (e.g., MIT).
Dataset follows its own license CC BY 4.0.
