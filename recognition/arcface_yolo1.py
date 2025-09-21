
# recognition/arcface_yolo.py
import os
import cv2
import csv
import time
import uuid
import functools
import numpy as np
import insightface
from ultralytics import YOLO

# ------------------------
# Shared stats storage (polled by /video_stats/)
# ------------------------
latest_stats = {"status": "no frames yet"}

# ------------------------
# Configuration
# ------------------------
YOLO_MODEL_PATH = "/home/ashres34/Desktop/SCI501/PPE_DATASET_YOLOv8_/runs/detect/yolov12-l-64-1002/weights/best.pt"
WORKERS_DIR = "/home/ashres34/Desktop/SCI501/worker_recognition/media/workers"
DB_PATH = os.path.join(os.path.dirname(__file__), "workers_db.npy")

# ArcFace config (quick & safe)
USE_GPU_ARCFACE = False          # set True if you have CUDA available
ARCFACE_DET_SIZE = (640, 640)    # detector input (larger => more accurate, slower)
ARCFACE_THRESH   = 0.5           # detector confidence (not ID threshold)

# ------------------------
# Models (singletons)
# ------------------------
yolo_model = YOLO(YOLO_MODEL_PATH)

@functools.lru_cache(maxsize=1)
def get_arcface():
    """
    Use buffalo_l (already cached on your machine) and memoize the instance so
    Django's autoreloader doesn't double-initialize or trigger downloads.
    """
    providers = ["CUDAExecutionProvider"] if USE_GPU_ARCFACE else ["CPUExecutionProvider"]
    ctx_id = 0 if USE_GPU_ARCFACE else -1
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=ARCFACE_DET_SIZE, det_thresh=ARCFACE_THRESH)
    print("[ArcFace] Loaded buffalo_l with providers:", providers, "ctx_id:", ctx_id)
    return app

arcface_model = get_arcface()

# ------------------------
# Session-wide accumulators & CSV logging
# ------------------------
_session_id = None
_session_started_ts = None
_csv_fh = None
_csv_writer = None

# Cumulative metrics across the current stream
_accum = {
    "frames": 0,
    "faces_total": 0,
    "recognized_total": 0,
    "unknown_total": 0,
    "avg_conf_overall": 0.0,   # running average over recognized faces
    "per_worker": {}           # {name: {"seen": int, "recognized": int, "avg_conf": float}}
}

def _rolling_avg(old_avg, old_n, new_value):
    # numerically stable running average
    return (old_avg * old_n + new_value) / max(1, old_n + 1)

def _reset_session(log_dir="/tmp"):
    """Reset accumulators and (re)open CSV log for a fresh run."""
    global _session_id, _session_started_ts, _accum, _csv_fh, _csv_writer
    _session_id = uuid.uuid4().hex[:8]
    _session_started_ts = time.time()
    _accum = {
        "frames": 0,
        "faces_total": 0,
        "recognized_total": 0,
        "unknown_total": 0,
        "avg_conf_overall": 0.0,
        "per_worker": {}
    }
    os.makedirs(log_dir, exist_ok=True)
    if _csv_fh:
        try:
            _csv_fh.close()
        except Exception:
            pass
    csv_path = os.path.join(log_dir, f"face_recog_{_session_id}.csv")
    _csv_fh = open(csv_path, "w", newline="")
    _csv_writer = csv.writer(_csv_fh)
    _csv_writer.writerow([
        "ts","session_id","frame_idx",
        "faces_in_frame","recognized_flag","unknown_flag",
        "worker","confidence","x1","y1","x2","y2"
    ])
    return csv_path

# ------------------------
# Worker Embeddings DB
# ------------------------
def build_worker_db(workers_dir=WORKERS_DIR, save_path=DB_PATH):
    """
    Builds/updates an embedding DB as {worker_name: mean_normalized_embedding}.
    Normalization -> faster cosine similarity at runtime.
    """
    db = {}
    for worker_name in os.listdir(workers_dir):
        worker_path = os.path.join(workers_dir, worker_name)
        if not os.path.isdir(worker_path):
            continue

        embeddings = []
        for fname in os.listdir(worker_path):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = os.path.join(worker_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = arcface_model.get(img)
            if faces:
                emb = faces[0].embedding.astype(np.float32)
                n = np.linalg.norm(emb) + 1e-9
                embeddings.append(emb / n)  # normalize per image

        if embeddings:
            mean = np.mean(embeddings, axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-9)  # normalize mean
            db[worker_name] = mean
            print(f"Added {worker_name} with {len(embeddings)} images")
        else:
            print(f"Warning: No faces found for {worker_name}")

    np.save(save_path, db)
    return db

def load_worker_db(db_path=DB_PATH):
    if not os.path.exists(db_path):
        return {}
    return np.load(db_path, allow_pickle=True).item()

# ------------------------
# Recognition on Frames
# ------------------------
def process_frame(frame, db, threshold=0.35, frame_idx=0):
    """
    Runs ArcFace and YOLO on an unmodified copy of the input frame,
    draws overlays on a separate display image, and returns the display image + stats.
    """
    t0 = time.time()
    stats = {
        "faces": 0,
        "recognized": 0,
        "unknown": 0,
        "recognized_workers": [],
        "yolo_detections": {},
        "timing_ms": {}
    }

    # Keep a pristine copy for inference, and a separate one for drawing
    raw = frame                     # don't touch this for inference
    display = frame.copy()          # draw on this only

    # ---------- Face recognition (on raw) ----------
    faces = arcface_model.get(raw)
    stats["faces"] = len(faces)

    global _accum, _csv_writer
    now = time.time()
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        emb = face.embedding.astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)

        best_match, best_score = None, -1.0
        for worker, ref_emb in db.items():
            sim = float(np.dot(ref_emb, emb))
            if sim > best_score:
                best_score = sim
                best_match = worker

        if best_score >= threshold and best_match is not None:
            label = f"{best_match} ({best_score:.2f})"
            color = (0, 255, 0)
            stats["recognized"] += 1
            stats["recognized_workers"].append(best_match)

            prev_n = _accum["recognized_total"]
            _accum["avg_conf_overall"] = _rolling_avg(_accum["avg_conf_overall"], prev_n, best_score)
            _accum["recognized_total"] += 1

            w = _accum["per_worker"].setdefault(best_match, {"seen": 0, "recognized": 0, "avg_conf": 0.0})
            w["seen"] += 1
            w["recognized"] += 1
            w["avg_conf"] = _rolling_avg(w["avg_conf"], w["recognized"] - 1, best_score)

            if _csv_writer:
                _csv_writer.writerow([now, _session_id, frame_idx, len(faces), 1, 0,
                                      best_match, round(best_score, 4), x1, y1, x2, y2])
        else:
            label = f"Unknown ({best_score:.2f})"
            color = (0, 0, 255)
            stats["unknown"] += 1

            if _csv_writer:
                _csv_writer.writerow([now, _session_id, frame_idx, len(faces), 0, 1,
                                      "Unknown", round(best_score, 4), x1, y1, x2, y2])

        # draw on the display image only
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    t1 = time.time()

    # ---------- YOLO detection (on raw) ----------
    detections = yolo_model(raw)[0]
    for box, cls_id in zip(detections.boxes.xyxy, detections.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls_id)
        class_name = detections.names[cls_id]

        stats["yolo_detections"][class_name] = stats["yolo_detections"].get(class_name, 0) + 1

        # draw on the display image only
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(display, class_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    t2 = time.time()
    stats["timing_ms"] = {
        "arcface": round((t1 - t0) * 1000, 2),
        "yolo":    round((t2 - t1) * 1000, 2),
        "total":   round((t2 - t0) * 1000, 2),
    }

    _accum["frames"] += 1
    _accum["faces_total"] += len(faces)
    _accum["unknown_total"] += stats["unknown"]

    # return the annotated display image
    return display, stats

# ------------------------
# Streaming Generator
# ------------------------
def recognize_from_video_stream(video_path=None, threshold=0.35, log_dir="/tmp"):
    """
    Yields MJPEG frames; updates 'latest_stats' with per-frame + cumulative metrics.
    Also writes a CSV log to /tmp/face_recog_<session>.csv.
    """
    global latest_stats

    csv_path = _reset_session(log_dir=log_dir)
    db = load_worker_db(DB_PATH)
    cap = cv2.VideoCapture(video_path if video_path else 0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, stats = process_frame(frame, db, threshold, frame_idx=frame_idx)
        frame_idx += 1

        # Assemble the payload polled by /video_stats/
        latest_stats = {
            "session_id": _session_id,
            "since": _session_started_ts,
            "frame_idx": frame_idx,
            "inst": stats,  # per-frame snapshot
            "cum": {
                "frames": _accum["frames"],
                "faces_total": _accum["faces_total"],
                "recognized_total": _accum["recognized_total"],
                "unknown_total": _accum["unknown_total"],
                "avg_conf_overall": round(_accum["avg_conf_overall"], 4),
                "recognition_rate": round(
                    _accum["recognized_total"] / max(1, _accum["faces_total"]), 4
                ),
                "per_worker": {
                    w: {
                        "seen": d["seen"],
                        "recognized": d["recognized"],
                        "avg_conf": round(d["avg_conf"], 4),
                        "recognition_rate": round(d["recognized"] / max(1, d["seen"]), 4),
                    }
                    for w, d in _accum["per_worker"].items()
                },
            },
            "log_csv": csv_path,
        }

        # Encode as JPEG
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()
    if _csv_fh:
        _csv_fh.flush()
