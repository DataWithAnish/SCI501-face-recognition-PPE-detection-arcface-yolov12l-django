# recognition/arcface_yolo.py
import os
import cv2
import csv
import time
import uuid
import json
import functools
import numpy as np
import insightface
from ultralytics import YOLO
from datetime import datetime

# ------------------------
# Shared stats storage (polled by /video_stats/)
# ------------------------
latest_stats = {"status": "no frames yet"}

# ------------------------
# Configuration
# ------------------------
YOLO_MODEL_PATH = "/Users/anishkumarshrestha/Desktop/PIE detector/SCI501/recognition/best.pt"
WORKERS_DIR = "/Users/anishkumarshrestha/Desktop/PIE detector/SCI501/media/workers"
DB_PATH = os.path.join(os.path.dirname(__file__), "workers_db.npy")

# Where to store all session artifacts
LOG_BASE_DIR = "/Users/anishkumarshrestha/Desktop/PIE detector/SCI501/recognition"

# ArcFace config (quick & safe)
USE_GPU_ARCFACE = False          # set True if you have CUDA available
ARCFACE_DET_SIZE = (640, 640)    # detector input (larger => more accurate, slower)
ARCFACE_THRESH   = 0.5           # detector confidence (not ID threshold)

# Performance knobs
DETECT_EVERY_N   = 5             # run ArcFace detector every N frames (tracks in between)
EMBED_EVERY_M    = 30            # refresh embedding for a live track every M frames
YOLO_EVERY_N     = 8             # run YOLO every N frames
TRACKER_TYPE     = "MOSSE"       # "MOSSE" (fastest), "KCF" (balanced), "CSRT" (robust) — auto-fallback if unavailable
MIN_FACE_SIZE    = 32            # skip tiny face boxes (short side < this many px)
NMS_IOU_MATCH    = 0.4           # match detector->tracker by IoU threshold
TOPK_TO_LOG      = 3             # how many nearest candidates to log per evaluation

# Optional: cap OpenCV threads if CPU thrashes (uncomment to tune)
# cv2.setNumThreads(4)

# ------------------------
# Models (singletons)
# ------------------------
yolo_model = YOLO(YOLO_MODEL_PATH)

@functools.lru_cache(maxsize=1)
def get_arcface():
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
_session_dir = None

_csv_fh = None           # minimal legacy CSV (kept for your UI)
_csv_writer = None

_pred_csv_fh = None      # detailed predictions CSV
_pred_csv_writer = None

_perf_csv_fh = None      # per-frame perf CSV
_perf_csv_writer = None

_yolo_csv_fh = None      # YOLO detections CSV
_yolo_csv_writer = None

# Cumulative metrics across the current stream
_accum = {
    "frames": 0,
    "faces_total": 0,
    "recognized_total": 0,
    "unknown_total": 0,
    "avg_conf_overall": 0.0,
    "per_worker": {}           # {name: {"seen": int, "recognized": int, "avg_conf": float}}
}

# Extra accumulators for summary.json
_perf_sums = {"arcface": 0.0, "yolo": 0.0, "total": 0.0}
_conf_recognized = []    # list of conf values when recognized
_conf_unknown = []       # list of conf values when below threshold (Unknown)
_yolo_class_totals = {}  # {class: count}

def _rolling_avg(old_avg, old_n, new_value):
    return (old_avg * old_n + new_value) / max(1, old_n + 1)

# ------------------------
# Tracking utilities
# ------------------------
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    return inter / max(1.0, area_a + area_b - inter)

# --- Lightweight fallback tracker: works without opencv-contrib ---
class LKBoxTracker:
    """
    Minimal bbox tracker using Lucas–Kanade optical flow.
    Has the same .init/.update API as cv2 trackers.
    """
    def __init__(self):
        self.prev_gray = None
        self.points = None
        self.box = None  # (x, y, w, h)

    def init(self, frame, rect):
        x, y, w, h = [int(v) for v in rect]
        self.box = (x, y, w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Feature points inside the box
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            self.points = None
            self.prev_gray = gray
            return True
        pts = cv2.goodFeaturesToTrack(
            roi, maxCorners=60, qualityLevel=0.01, minDistance=5, blockSize=7
        )
        if pts is None:
            self.points = None
            self.prev_gray = gray
            return True
        pts = pts.reshape(-1, 2)
        # shift to full-image coords
        pts[:, 0] += x
        pts[:, 1] += y
        self.points = pts.astype(np.float32)
        self.prev_gray = gray
        return True

    def update(self, frame):
        if self.box is None:
            return False, (0, 0, 0, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If we lost feature points, reinitialize around the last box
        if self.points is None or len(self.points) < 12:
            self.init(frame, self.box)

        if self.points is None or len(self.points) < 4:
            self.prev_gray = gray
            return False, self.box

        next_pts, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        st = st.reshape(-1)
        good_new = next_pts[st == 1] if next_pts is not None else None

        if good_new is None or len(good_new) < 6:
            self.prev_gray = gray
            self.points = None
            return False, self.box

        # Update bbox from new point cloud (min/max bounds with a small margin)
        min_xy = np.percentile(good_new, 5, axis=0)
        max_xy = np.percentile(good_new, 95, axis=0)
        x1, y1 = min_xy
        x2, y2 = max_xy
        pad_x = 0.05 * (x2 - x1 + 1)
        pad_y = 0.05 * (y2 - y1 + 1)
        x1 = int(max(0, x1 - pad_x)); y1 = int(max(0, y1 - pad_y))
        x2 = int(min(frame.shape[1]-1, x2 + pad_x)); y2 = int(min(frame.shape[0]-1, y2 + pad_y))
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        self.box = (x1, y1, w, h)

        # Keep tracking points & prev frame
        self.prev_gray = gray
        self.points = good_new.astype(np.float32)

        return True, self.box

def _create_tracker():
    """
    Return a tracker instance compatible with .init(frame, (x,y,w,h)) and .update(frame)->(ok,(x,y,w,h)).
    Tries OpenCV trackers across versions; if none exist, falls back to LKBoxTracker.
    """
    candidates = []

    # Respect user preference first
    t = (TRACKER_TYPE or "MOSSE").upper()

    # Helper to resolve dotted names like "legacy.TrackerMOSSE_create"
    def resolve_factory(dotted):
        mod = cv2
        for part in dotted.split("."):
            if not hasattr(mod, part):
                return None
            mod = getattr(mod, part)
        return mod if callable(mod) else None

    name_map = {
        "CSRT": ["legacy.TrackerCSRT_create", "TrackerCSRT_create"],
        "KCF":  ["legacy.TrackerKCF_create",  "TrackerKCF_create"],
        "MOSSE":["legacy.TrackerMOSSE_create","TrackerMOSSE_create"],
        "MIL":  ["legacy.TrackerMIL_create",  "TrackerMIL_create"],
        "MEDIANFLOW": ["legacy.TrackerMedianFlow_create", "TrackerMedianFlow_create"],
        "TLD":  ["legacy.TrackerTLD_create",  "TrackerTLD_create"],
    }

    # Add the requested type first (if present)
    if t in name_map:
        for dotted in name_map[t]:
            f = resolve_factory(dotted)
            if f: candidates.append(f)

    # Then, add a fallback list in order of preference
    for key in ["MOSSE", "KCF", "CSRT", "MIL", "MEDIANFLOW"]:
        if key == t:
            continue
        for dotted in name_map[key]:
            f = resolve_factory(dotted)
            if f: candidates.append(f)

    # Try to create a real cv2 tracker
    for factory in candidates:
        try:
            return factory()
        except Exception:
            continue

    # Last resort: LK optical-flow tracker (no contrib required)
    return LKBoxTracker()

class FaceTrack:
    __slots__ = ("id", "tracker", "bbox", "label", "conf", "last_emb", "last_refresh_idx", "missed")
    def __init__(self, tid, frame, bbox_xyxy):
        self.id = tid
        self.tracker = _create_tracker()
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        self.bbox = (x1, y1, x2, y2)
        self.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        self.label = "Unknown"
        self.conf = -1.0
        self.last_emb = None
        self.last_refresh_idx = -10**9
        self.missed = 0

    def update(self, frame):
        ok, box = self.tracker.update(frame)
        if not ok:
            self.missed += 1
            return False
        x, y, w, h = box
        self.bbox = (int(x), int(y), int(x + w), int(y + h))
        self.missed = 0
        return True

class FaceTrackManager:
    def __init__(self):
        self._tracks = {}
        self._next_id = 1

    def match_and_update(self, frame, detected_boxes):
        """
        Match detector boxes to existing tracks by IoU; update trackers in place.
        Return list of active tracks.
        """
        # Update existing trackers first
        active_ids = []
        for tid, tr in list(self._tracks.items()):
            if tr.update(frame):
                active_ids.append(tid)
            else:
                if tr.missed > 5:
                    del self._tracks[tid]

        # Greedy IoU matching
        unmatched = []
        for det in detected_boxes:
            best_iou, best_tid = 0.0, None
            for tid in active_ids:
                iou = _iou(self._tracks[tid].bbox, det)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_iou >= NMS_IOU_MATCH and best_tid is not None:
                # re-init tracker on better box to reduce drift
                tr = self._tracks[best_tid]
                x1, y1, x2, y2 = map(int, det)
                tr.tracker = _create_tracker()
                tr.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                tr.bbox = (x1, y1, x2, y2)
            else:
                unmatched.append(det)

        # New tracks for unmatched detections
        for det in unmatched:
            tid = self._next_id; self._next_id += 1
            self._tracks[tid] = FaceTrack(tid, frame, det)

        return list(self._tracks.values())

    def update_only(self, frame):
        for tid, tr in list(self._tracks.items()):
            if not tr.update(frame) and tr.missed > 5:
                del self._tracks[tid]
        return list(self._tracks.values())

    def all(self):
        return list(self._tracks.values())

# Global trackers & caches (reset per session)
_track_mgr = FaceTrackManager()
_prev_yolo = {"frame_idx": -10**9, "detections": None}

def _ensure_dirs(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    ses_root = os.path.join(base_dir, "sessions")
    os.makedirs(ses_root, exist_ok=True)
    return ses_root

def _reset_session(log_dir=None):
    """
    Prepare a fresh session: accumulators, trackers, file outputs.
    Returns csv_path for legacy UI.
    """
    global _session_id, _session_started_ts, _session_dir
    global _accum, _csv_fh, _csv_writer, _pred_csv_fh, _pred_csv_writer
    global _perf_csv_fh, _perf_csv_writer, _yolo_csv_fh, _yolo_csv_writer
    global _track_mgr, _prev_yolo, _perf_sums, _conf_recognized, _conf_unknown, _yolo_class_totals

    _session_id = uuid.uuid4().hex[:8]
    _session_started_ts = time.time()

    _accum = {"frames": 0, "faces_total": 0, "recognized_total": 0, "unknown_total": 0,
              "avg_conf_overall": 0.0, "per_worker": {}}

    _perf_sums = {"arcface": 0.0, "yolo": 0.0, "total": 0.0}
    _conf_recognized = []
    _conf_unknown = []
    _yolo_class_totals = {}

    # reset trackers & caches
    _track_mgr = FaceTrackManager()
    _prev_yolo = {"frame_idx": -10**9, "detections": None}

    # Directories
    ses_root = _ensure_dirs(log_dir or LOG_BASE_DIR)
    _session_dir = os.path.join(ses_root, _session_id)
    os.makedirs(_session_dir, exist_ok=True)

    # --- Legacy per-evaluation CSV (keep schema for your UI) ---
    csv_path = os.path.join(_session_dir, f"face_recog_{_session_id}.csv")
    if _csv_fh:
        try: _csv_fh.close()
        except Exception: pass
    _csv_fh = open(csv_path, "w", newline="")
    _csv_writer = csv.writer(_csv_fh)
    _csv_writer.writerow([
        "ts","session_id","frame_idx",
        "faces_in_frame","recognized_flag","unknown_flag",
        "worker","confidence","x1","y1","x2","y2"
    ])

    # --- Detailed predictions CSV ---
    pred_path = os.path.join(_session_dir, f"predictions_{_session_id}.csv")
    if _pred_csv_fh:
        try: _pred_csv_fh.close()
        except Exception: pass
    _pred_csv_fh = open(pred_path, "w", newline="")
    _pred_csv_writer = csv.writer(_pred_csv_fh)
    # top-k columns dynamically
    header = ["ts","session_id","frame_idx","track_id","x1","y1","x2","y2",
              "predicted","confidence","threshold","decision"]
    for k in range(1, TOPK_TO_LOG+1):
        header += [f"top{k}_name", f"top{k}_sim"]
    _pred_csv_writer.writerow(header)

    # --- Per-frame perf CSV ---
    perf_path = os.path.join(_session_dir, f"perf_{_session_id}.csv")
    if _perf_csv_fh:
        try: _perf_csv_fh.close()
        except Exception: pass
    _perf_csv_fh = open(perf_path, "w", newline="")
    _perf_csv_writer = csv.writer(_perf_csv_fh)
    _perf_csv_writer.writerow([
        "frame_idx","arcface_ms","yolo_ms","total_ms","fps",
        "tracks","known_now","unknown_now"
    ])

    # --- YOLO detections CSV ---
    yolo_path = os.path.join(_session_dir, f"yolo_{_session_id}.csv")
    if _yolo_csv_fh:
        try: _yolo_csv_fh.close()
        except Exception: pass
    _yolo_csv_fh = open(yolo_path, "w", newline="")
    _yolo_csv_writer = csv.writer(_yolo_csv_fh)
    _yolo_csv_writer.writerow(["frame_idx","class","x1","y1","x2","y2"])

    # Save a small run-config snapshot immediately
    config = {
        "session_id": _session_id,
        "started_iso": datetime.fromtimestamp(_session_started_ts).isoformat(),
        "models": {
            "yolo_model_path": YOLO_MODEL_PATH,
            "arcface": {"name": "buffalo_l", "det_size": ARCFACE_DET_SIZE, "det_thresh": ARCFACE_THRESH,
                        "gpu": USE_GPU_ARCFACE}
        },
        "performance_knobs": {
            "DETECT_EVERY_N": DETECT_EVERY_N,
            "EMBED_EVERY_M": EMBED_EVERY_M,
            "YOLO_EVERY_N": YOLO_EVERY_N,
            "TRACKER_TYPE": TRACKER_TYPE,
            "MIN_FACE_SIZE": MIN_FACE_SIZE,
            "NMS_IOU_MATCH": NMS_IOU_MATCH,
            "TOPK_TO_LOG": TOPK_TO_LOG
        }
    }
    with open(os.path.join(_session_dir, f"config_{_session_id}.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    return csv_path

# ------------------------
# Worker Embeddings DB
# ------------------------
def build_worker_db(workers_dir=WORKERS_DIR, save_path=DB_PATH):
    db = {}
    for worker_name in os.listdir(workers_dir):
        worker_path = os.path.join(workers_dir, worker_name)
        if not os.path.isdir(worker_path): continue
        embeddings = []
        for fname in os.listdir(worker_path):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")): continue
            img_path = os.path.join(worker_path, fname)
            img = cv2.imread(img_path)
            if img is None: continue
            faces = arcface_model.get(img)
            if faces:
                emb = faces[0].embedding.astype(np.float32)
                n = np.linalg.norm(emb) + 1e-9
                embeddings.append(emb / n)
        if embeddings:
            mean = np.mean(embeddings, axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-9)
            db[worker_name] = mean
            print(f"Added {worker_name} with {len(embeddings)} images")
        else:
            print(f"Warning: No faces found for {worker_name}")
    np.save(save_path, db)
    return db

def load_worker_db(db_path=DB_PATH):
    if not os.path.exists(db_path):
        return {}, None, None
    db = np.load(db_path, allow_pickle=True).item()
    names = np.array(list(db.keys()))
    if len(names) == 0:
        return {}, None, None
    mat = np.stack([db[n] for n in names]).astype(np.float32)  # (M, D)
    # L2-normalize rows to fix old, unnormalized DBs
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat = mat / norms
    return db, names, mat

# ------------------------
# Recognition helpers
# ------------------------
def _topk_embedding(emb_normed, names_arr, mat, k=3):
    """Return top-k (name, sim) sorted by similarity."""
    sims = mat @ emb_normed  # (M,)
    if sims.size == 0:
        return []
    k = int(min(k, sims.shape[0]))
    # partial sort then order
    idxs = np.argpartition(-sims, range(k))[:k]
    idxs = idxs[np.argsort(-sims[idxs])]
    return [(str(names_arr[i]), float(sims[i])) for i in idxs]

def _match_embedding(emb_normed, names_arr, mat):
    sims = mat @ emb_normed  # (M,)
    idx = int(np.argmax(sims))
    return names_arr[idx], float(sims[idx])

# ------------------------
# Recognition on Frames (with tracking & duty-cycling)
# ------------------------
def process_frame(frame, db_pack, threshold=0.35, frame_idx=0):
    """
    Faster pipeline:
    - Every DETECT_EVERY_N frames: run ArcFace detector, (re)initialize trackers, compute embeddings & IDs.
    - Other frames: update trackers only; reuse last IDs; refresh embedding sparsely (EMBED_EVERY_M).
    - YOLO is run every YOLO_EVERY_N frames; else reuse.
    """
    db_dict, db_names, db_mat = db_pack
    t0 = time.time()
    stats = {
        "faces": 0,
        "recognized": 0,     # will be set to "known_now" for UI truthfulness
        "unknown": 0,
        "recognized_workers": [],
        "yolo_detections": {},
        "timing_ms": {}
    }

    raw = frame
    display = frame.copy()

    global _accum, _csv_writer, _pred_csv_writer, _prev_yolo
    global _perf_sums, _conf_recognized, _conf_unknown, _yolo_csv_writer, _yolo_class_totals

    # --------- Face detect vs track ----------
    detect_frame = (frame_idx % DETECT_EVERY_N == 0)
    faces = []
    if detect_frame:
        faces = arcface_model.get(raw)  # returns list with .bbox and .embedding available
        det_boxes = []
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            if min(x2-x1, y2-y1) < MIN_FACE_SIZE:
                continue
            det_boxes.append((x1, y1, x2, y2))
        tracks = _track_mgr.match_and_update(raw, det_boxes)
    else:
        tracks = _track_mgr.update_only(raw)

    # For metrics, "faces in frame" ~= current number of active tracks
    stats["faces"] = len(tracks)
    now = time.time()

    # Compute/refresh identity where needed
    for tr in tracks:
        x1, y1, x2, y2 = tr.bbox
        # refresh embedding on detect frames OR every EMBED_EVERY_M frames to limit drift
        need_refresh = detect_frame or (frame_idx - tr.last_refresh_idx >= EMBED_EVERY_M)
        if need_refresh:
            # Crop safely
            x1c = max(0, x1); y1c = max(0, y1); x2c = min(display.shape[1]-1, x2); y2c = min(display.shape[0]-1, y2)
            crop = raw[y1c:y2c, x1c:x2c]
            if crop.size > 0:
                emb = None
                if detect_frame:
                    # Find matching full-frame face by IoU to reuse its embedding
                    best_i, best_iou = -1, 0.0
                    for i, f in enumerate(faces):
                        fb = tuple(map(int, f.bbox))
                        iou = _iou(fb, tr.bbox)
                        if iou > best_iou:
                            best_i, best_iou = i, iou
                    if best_i >= 0:
                        emb = faces[best_i].embedding.astype(np.float32)
                if emb is None:
                    # Fallback: detect-from-crop (cheap ROI)
                    f2 = arcface_model.get(crop)
                    if f2:
                        emb = f2[0].embedding.astype(np.float32)

                if emb is not None:
                    emb /= (np.linalg.norm(emb) + 1e-9)
                    tr.last_emb = emb
                    tr.last_refresh_idx = frame_idx

                    # Vectorized match + top-k if DB available
                    if db_mat is not None and db_names is not None and db_mat.size > 0:
                        topk = _topk_embedding(emb, db_names, db_mat, TOPK_TO_LOG)
                        best_match, best_score = topk[0] if len(topk) else (None, -1.0)
                    else:
                        topk = []
                        best_match, best_score = None, -1.0

                    if best_match is not None and best_score >= threshold:
                        tr.label = best_match
                        tr.conf = best_score
                        # cumulative stats
                        stats["recognized_workers"].append(tr.label)
                        prev_n = _accum["recognized_total"]
                        _accum["avg_conf_overall"] = _rolling_avg(_accum["avg_conf_overall"], prev_n, tr.conf)
                        _accum["recognized_total"] += 1
                        w = _accum["per_worker"].setdefault(tr.label, {"seen": 0, "recognized": 0, "avg_conf": 0.0})
                        w["seen"] += 1; w["recognized"] += 1
                        w["avg_conf"] = _rolling_avg(w["avg_conf"], w["recognized"] - 1, tr.conf)
                        _conf_recognized.append(tr.conf)

                        # legacy CSV (per evaluation)
                        if _csv_writer:
                            _csv_writer.writerow([now, _session_id, frame_idx, len(tracks), 1, 0,
                                                  tr.label, round(tr.conf, 4), x1, y1, x2, y2])
                        # detailed predictions CSV
                        if _pred_csv_writer:
                            row = [now, _session_id, frame_idx, tr.id, x1, y1, x2, y2,
                                   tr.label, round(tr.conf, 6), threshold, "recognized"]
                            for k in range(TOPK_TO_LOG):
                                if k < len(topk):
                                    row += [topk[k][0], round(topk[k][1], 6)]
                                else:
                                    row += ["", ""]
                            _pred_csv_writer.writerow(row)

                    else:
                        # Unknown decision
                        tr.label = "Unknown"
                        tr.conf = best_score if best_score is not None else -1.0
                        _accum["unknown_total"] += 1
                        _conf_unknown.append(tr.conf)
                        if _csv_writer:
                            _csv_writer.writerow([now, _session_id, frame_idx, len(tracks), 0, 1,
                                                  "Unknown", round(tr.conf, 4), x1, y1, x2, y2])
                        if _pred_csv_writer:
                            row = [now, _session_id, frame_idx, tr.id, x1, y1, x2, y2,
                                   "Unknown", round(tr.conf, 6), threshold, "unknown"]
                            for k in range(TOPK_TO_LOG):
                                if k < len(topk):
                                    row += [topk[k][0], round(topk[k][1], 6)]
                                else:
                                    row += ["", ""]
                            _pred_csv_writer.writerow(row)

        # Draw current track box + label (always on display)
        color = (0, 255, 0) if tr.label != "Unknown" else (0, 0, 255)
        label = f"{tr.label} ({tr.conf:.2f})" if tr.conf is not None else tr.label
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    t1 = time.time()

    # --- Instant known/unknown for UI & perf log ---
    known_now = sum(1 for tr in tracks if tr.label != "Unknown")
    stats["recognized"] = known_now
    stats["unknown"] = max(0, len(tracks) - known_now)

    # ---------- YOLO detection (duty-cycled) ----------
    if frame_idx - _prev_yolo["frame_idx"] >= YOLO_EVERY_N:
        det = yolo_model(raw)[0]
        yolo_stats = {}
        yolo_boxes = []
        for box, cls_id in zip(det.boxes.xyxy, det.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls_id)
            class_name = det.names[cls_id]
            yolo_stats[class_name] = yolo_stats.get(class_name, 0) + 1
            yolo_boxes.append((x1, y1, x2, y2, class_name))
            # log row
            if _yolo_csv_writer:
                _yolo_csv_writer.writerow([frame_idx, class_name, x1, y1, x2, y2])
        _prev_yolo = {"frame_idx": frame_idx, "detections": (yolo_stats, yolo_boxes)}
        # accumulate totals
        for k, v in yolo_stats.items():
            _yolo_class_totals[k] = _yolo_class_totals.get(k, 0) + v
    else:
        yolo_stats, yolo_boxes = _prev_yolo["detections"] if _prev_yolo["detections"] else ({}, [])

    # Draw YOLO cached detections
    for (x1, y1, x2, y2, class_name) in yolo_boxes:
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(display, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    stats["yolo_detections"] = yolo_stats

    t2 = time.time()
    arc_ms = (t1 - t0) * 1000.0
    yolo_ms = (t2 - t1) * 1000.0
    tot_ms = (t2 - t0) * 1000.0
    fps = 1000.0 / max(1e-6, tot_ms)
    stats["timing_ms"] = {
        "arcface": round(arc_ms, 2),
        "yolo":    round(yolo_ms, 2),
        "total":   round(tot_ms, 2),
    }

    # perf CSV
    if _perf_csv_writer:
        _perf_csv_writer.writerow([
            frame_idx, round(arc_ms,2), round(yolo_ms,2), round(tot_ms,2), round(fps,2),
            len(tracks), known_now, max(0, len(tracks)-known_now)
        ])

    # cumulative
    _accum["frames"] += 1
    _accum["faces_total"] += len(tracks)
    _perf_sums["arcface"] += arc_ms
    _perf_sums["yolo"] += yolo_ms
    _perf_sums["total"] += tot_ms

    return display, stats

# -------- Summary / Accuracy --------
def _maybe_load_ground_truth(gt_path):
    """
    Optional: ground_truth.csv with columns:
    frame_idx,track_id,true_worker
    """
    if not os.path.exists(gt_path):
        return None
    gt = {}
    with open(gt_path, "r", newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            try:
                fi = int(row["frame_idx"])
                tid = int(row["track_id"])
                lbl = str(row["true_worker"]).strip()
                gt[(fi, tid)] = lbl
            except Exception:
                continue
    return gt

def _compute_accuracy_from_gt(pred_csv_path, gt_map):
    """
    Compute accuracy/PR/F1 using predictions_<session>.csv and a gt map.
    Returns dict with metrics (overall + per-class) & confusion matrix.
    """
    if gt_map is None:
        return None

    # Counters
    classes = set(gt_map.values())
    TP = {c:0 for c in classes}
    FP = {c:0 for c in classes}
    FN = {c:0 for c in classes}
    conf_matrix = {}  # true -> {pred: count}

    matched = 0

    with open(pred_csv_path, "r", newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            fi = int(row["frame_idx"]); tid = int(row["track_id"])
            decision = row["decision"]
            pred = row["predicted"].strip()
            if (fi, tid) not in gt_map:
                continue
            matched += 1
            true = gt_map[(fi, tid)]
            if decision == "recognized":
                if pred == true:
                    TP[true] = TP.get(true,0) + 1
                else:
                    FP[pred] = FP.get(pred,0) + 1
                    FN[true] = FN.get(true,0) + 1
                # confusion matrix
                conf_matrix.setdefault(true, {})
                conf_matrix[true][pred] = conf_matrix[true].get(pred, 0) + 1
            else:
                # unknown counts as a miss on the true class
                FN[true] = FN.get(true,0) + 1
                conf_matrix.setdefault(true, {})
                conf_matrix[true]["Unknown"] = conf_matrix[true].get("Unknown", 0) + 1

    # Per-class metrics
    per_class = {}
    for c in sorted(set(list(TP.keys()) + list(FP.keys()) + list(FN.keys()))):
        tp, fp, fn = TP.get(c,0), FP.get(c,0), FN.get(c,0)
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = (2*prec*rec) / max(1e-9, (prec+rec))
        per_class[c] = {"precision": round(prec,4), "recall": round(rec,4), "f1": round(f1,4), "tp": tp, "fp": fp, "fn": fn}

    # Overall
    total_tp = sum(TP.values())
    total = matched
    acc = total_tp / max(1, total)

    # Micro
    micro_prec = total_tp / max(1, total_tp + sum(FP.values()))
    micro_rec  = total_tp / max(1, total_tp + sum(FN.values()))
    micro_f1   = (2*micro_prec*micro_rec) / max(1e-9, (micro_prec+micro_rec))

    # Macro
    if per_class:
        macro_prec = np.mean([v["precision"] for v in per_class.values()])
        macro_rec  = np.mean([v["recall"] for v in per_class.values()])
        macro_f1   = np.mean([v["f1"] for v in per_class.values()])
    else:
        macro_prec = macro_rec = macro_f1 = 0.0

    return {
        "matched_pairs": matched,
        "overall_accuracy": round(acc,4),
        "micro": {"precision": round(micro_prec,4), "recall": round(micro_rec,4), "f1": round(micro_f1,4)},
        "macro": {"precision": round(macro_prec,4), "recall": round(macro_rec,4), "f1": round(macro_f1,4)},
        "per_class": per_class,
        "confusion_matrix": conf_matrix
    }

def _finalize_session(db_pack, threshold):
    """
    Write summary_<session>.json with totals, timing, histograms, config,
    and optional accuracy if ground_truth.csv is present.
    """
    global _csv_fh, _pred_csv_fh, _perf_csv_fh, _yolo_csv_fh
    try:
        if _csv_fh: _csv_fh.flush()
        if _pred_csv_fh: _pred_csv_fh.flush()
        if _perf_csv_fh: _perf_csv_fh.flush()
        if _yolo_csv_fh: _yolo_csv_fh.flush()
    except Exception:
        pass

    ended_ts = time.time()
    duration_s = ended_ts - _session_started_ts

    frames = max(1, _accum["frames"])
    avg_arc = _perf_sums["arcface"] / frames
    avg_yolo = _perf_sums["yolo"] / frames
    avg_total = _perf_sums["total"] / frames
    avg_fps = 1000.0 / max(1e-6, avg_total)

    # Confidence histograms
    def hist_data(vals, bins=np.linspace(-1.0, 1.0, 21)):
        if len(vals) == 0:
            return {"bins": list(map(float, bins)), "counts": [0]* (len(bins)-1)}
        counts, edges = np.histogram(np.array(vals, dtype=np.float32), bins=bins)
        return {"bins": [float(x) for x in edges], "counts": [int(x) for x in counts]}

    db_dict, db_names, db_mat = db_pack
    emb_dim = int(db_mat.shape[1]) if db_mat is not None else 0
    num_workers = int(len(db_names)) if db_names is not None else 0

    summary = {
        "session": {
            "id": _session_id,
            "started_iso": datetime.fromtimestamp(_session_started_ts).isoformat(),
            "ended_iso": datetime.fromtimestamp(ended_ts).isoformat(),
            "duration_sec": round(duration_s, 3),
            "dir": _session_dir
        },
        "totals": {
            "frames": _accum["frames"],
            "faces_total": _accum["faces_total"],
            "recognized_total": _accum["recognized_total"],
            "unknown_total": _accum["unknown_total"],
            "recognition_rate": round(_accum["recognized_total"] / max(1, _accum["faces_total"]), 4),
            "avg_conf_overall": round(_accum["avg_conf_overall"], 4),
            "per_worker": _accum["per_worker"],
        },
        "timing_avg_ms": {
            "arcface": round(avg_arc, 2),
            "yolo": round(avg_yolo, 2),
            "total": round(avg_total, 2),
            "fps_avg": round(avg_fps, 2),
        },
        "yolo_totals": _yolo_class_totals,
        "confidence_hist": {
            "recognized": hist_data(_conf_recognized),
            "unknown": hist_data(_conf_unknown),
        },
        "config": {
            "threshold": threshold,
            "DETECT_EVERY_N": DETECT_EVERY_N,
            "EMBED_EVERY_M": EMBED_EVERY_M,
            "YOLO_EVERY_N": YOLO_EVERY_N,
            "TRACKER_TYPE": TRACKER_TYPE,
            "MIN_FACE_SIZE": MIN_FACE_SIZE,
            "NMS_IOU_MATCH": NMS_IOU_MATCH,
            "ARCFACE_DET_SIZE": ARCFACE_DET_SIZE,
            "ARCFACE_THRESH": ARCFACE_THRESH,
            "USE_GPU_ARCFACE": USE_GPU_ARCFACE,
            "db_num_workers": num_workers,
            "db_embedding_dim": emb_dim,
            "yolo_model_path": YOLO_MODEL_PATH
        },
        "files": {
            "legacy_csv": os.path.join(_session_dir, f"face_recog_{_session_id}.csv"),
            "predictions_csv": os.path.join(_session_dir, f"predictions_{_session_id}.csv"),
            "perf_csv": os.path.join(_session_dir, f"perf_{_session_id}.csv"),
            "yolo_csv": os.path.join(_session_dir, f"yolo_{_session_id}.csv"),
            "config_json": os.path.join(_session_dir, f"config_{_session_id}.json"),
        }
    }

    # Optional accuracy with ground truth
    gt_path = os.path.join(_session_dir, "ground_truth.csv")
    acc = _maybe_load_ground_truth(gt_path)
    if acc is not None:
        preds_path = summary["files"]["predictions_csv"]
        metrics = _compute_accuracy_from_gt(preds_path, acc)
        summary["ground_truth_metrics"] = metrics
        summary["files"]["ground_truth_csv"] = gt_path

    # Write summary
    with open(os.path.join(_session_dir, f"summary_{_session_id}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

# ------------------------
# Streaming Generator
# ------------------------
def recognize_from_video_stream(video_path=None, threshold=0.35, log_dir=LOG_BASE_DIR):
    """
    Yields MJPEG frames; updates 'latest_stats' with per-frame + cumulative metrics.
    Also writes multiple CSVs and a summary JSON under LOG_BASE_DIR/sessions/<session_id>/.
    """
    global latest_stats

    csv_path = _reset_session(log_dir=log_dir)
    db_pack = load_worker_db(DB_PATH)
    cap = cv2.VideoCapture(video_path if video_path else 0)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, stats = process_frame(frame, db_pack, threshold, frame_idx=frame_idx)
            frame_idx += 1

            latest_stats = {
                "session_id": _session_id,
                "since": _session_started_ts,
                "frame_idx": frame_idx,
                "inst": stats,
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
                "log_dir": _session_dir,
            }

            # Encode as JPEG
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()
        # finalize write of summary & close files
        _finalize_session(db_pack, threshold)
        try:
            if _csv_fh: _csv_fh.close()
            if _pred_csv_fh: _pred_csv_fh.close()
            if _perf_csv_fh: _perf_csv_fh.close()
            if _yolo_csv_fh: _yolo_csv_fh.close()
        except Exception:
            pass
