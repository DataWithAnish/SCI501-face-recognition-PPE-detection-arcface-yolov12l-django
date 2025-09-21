# recognition/arcface_yolo.py
import os
import sys
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
YOLO_MODEL_PATH = "/home/ashres34/Desktop/SCI501/PPE_DATASET_YOLOv8_/runs/detect/yolov12-l-64-1002/weights/best.pt"
WORKERS_DIR = "/home/ashres34/Desktop/SCI501/worker_recognition/media/workers"
DB_PATH = os.path.join(os.path.dirname(__file__), "workers_db.npy")

# Where to store all session artifacts
LOG_BASE_DIR = "/home/ashres34/Desktop/SCI501/worker_recognition/recognition"

# ArcFace config
USE_GPU_ARCFACE = False
ARCFACE_DET_SIZE = (640, 640)
ARCFACE_THRESH   = 0.5   # detector confidence (not ID threshold)

# Performance & tracking knobs (tuned for CPU laptops)
DETECT_EVERY_N   = 2     # re-detect faces more often to avoid drift
EMBED_EVERY_M    = 15
YOLO_EVERY_N     = 8
TRACKER_TYPE     = "CSRT"  # more robust than MOSSE/KCF on CPU
MIN_FACE_SIZE    = 36
NMS_IOU_MATCH    = 0.5    # IoU to match det->track (was 0.4)

# Detection & track quality controls
FACE_DET_MIN_SCORE = 0.60   # drop weak face detections
FACE_DET_NMS_IOU   = 0.45   # NMS among face detections

# De-dup & suppression
TRACK_DEDUP_IOU    = 0.55   # merge tracks that overlap a lot (more aggressive than 0.70)
TRACK_MAX_MISSED   = 8      # frames to keep a lost track
UNKNOWN_SUPPRESS_IOU = 0.50 # hide Unknown track if it overlaps a kept Known track

# YOLO drawing options to avoid visual “duplicates”
YOLO_EXCLUDE_CLASSES = {"person"}  # case-insensitive check

TOPK_TO_LOG      = 3
# Optional: cap OpenCV threads if CPU thrashes
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

_csv_fh = None; _csv_writer = None
_pred_csv_fh = None; _pred_csv_writer = None
_perf_csv_fh = None; _perf_csv_writer = None
_yolo_csv_fh = None; _yolo_csv_writer = None

_accum = {
    "frames": 0, "faces_total": 0, "recognized_total": 0,
    "unknown_total": 0, "avg_conf_overall": 0.0, "per_worker": {}
}

_perf_sums = {"arcface": 0.0, "yolo": 0.0, "total": 0.0}
_conf_recognized = []
_conf_unknown = []
_yolo_class_totals = {}

def _rolling_avg(old_avg, old_n, new_value):
    return (old_avg * old_n + new_value) / max(1, old_n + 1)

# ------------------------
# Geometry helpers
# ------------------------
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    return inter / max(1.0, area_a + area_b - inter)

def _area(b):
    x1, y1, x2, y2 = b
    return max(1, (x2 - x1)) * max(1, (y2 - y1))

def _nms_indices(boxes, scores, iou_thr):
    """Return indices to keep after greedy NMS."""
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        suppress = []
        for k, j in enumerate(rest):
            if _iou(tuple(boxes[i]), tuple(boxes[int(j)])) >= iou_thr:
                suppress.append(k)
        if suppress:
            mask = np.ones(rest.shape[0], dtype=bool)
            mask[suppress] = False
            order = np.concatenate([[order[0]], rest[mask]])
        order = order[1:]
    return keep

# ------------------------
# Tracking utilities
# ------------------------
class LKBoxTracker:
    """Minimal bbox tracker using Lucas–Kanade optical flow."""
    def __init__(self):
        self.prev_gray = None
        self.points = None
        self.box = None  # (x, y, w, h)

    def init(self, frame, rect):
        x, y, w, h = [int(v) for v in rect]
        self.box = (x, y, w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            self.points = None; self.prev_gray = gray; return True
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=60, qualityLevel=0.01,
                                      minDistance=5, blockSize=7)
        if pts is None:
            self.points = None; self.prev_gray = gray; return True
        pts = pts.reshape(-1, 2)
        pts[:, 0] += x; pts[:, 1] += y
        self.points = pts.astype(np.float32)
        self.prev_gray = gray
        return True

    def update(self, frame):
        if self.box is None:
            return False, (0, 0, 0, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.points is None or len(self.points) < 12:
            self.init(frame, self.box)
        if self.points is None or len(self.points) < 4:
            self.prev_gray = gray
            return False, self.box

        next_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        st = st.reshape(-1)
        good_new = next_pts[st == 1] if next_pts is not None else None
        if good_new is None or len(good_new) < 6:
            self.prev_gray = gray; self.points = None
            return False, self.box

        min_xy = np.percentile(good_new, 5, axis=0)
        max_xy = np.percentile(good_new, 95, axis=0)
        x1, y1 = min_xy; x2, y2 = max_xy
        pad_x = 0.05 * (x2 - x1 + 1); pad_y = 0.05 * (y2 - y1 + 1)
        x1 = int(max(0, x1 - pad_x)); y1 = int(max(0, y1 - pad_y))
        x2 = int(min(frame.shape[1]-1, x2 + pad_x)); y2 = int(min(frame.shape[0]-1, y2 + pad_y))
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        self.box = (x1, y1, w, h)
        self.prev_gray = gray
        self.points = good_new.astype(np.float32)
        return True, self.box

def _create_tracker():
    """Prefer CSRT>KCF>MOSSE; fallback to LK if contrib not present."""
    def resolve_factory(dotted):
        mod = cv2
        for part in dotted.split("."):
            if not hasattr(mod, part): return None
            mod = getattr(mod, part)
        return mod if callable(mod) else None

    t = (TRACKER_TYPE or "CSRT").upper()
    name_map = {
        "CSRT": ["legacy.TrackerCSRT_create", "TrackerCSRT_create"],
        "KCF":  ["legacy.TrackerKCF_create",  "TrackerKCF_create"],
        "MOSSE":["legacy.TrackerMOSSE_create","TrackerMOSSE_create"],
        "MIL":  ["legacy.TrackerMIL_create",  "TrackerMIL_create"],
        "MEDIANFLOW": ["legacy.TrackerMedianFlow_create", "TrackerMedianFlow_create"],
    }
    candidates = []
    if t in name_map:
        for dotted in name_map[t]:
            f = resolve_factory(dotted)
            if f: candidates.append(f)
    for key in ["CSRT", "KCF", "MOSSE", "MIL", "MEDIANFLOW"]:
        if key == t: continue
        for dotted in name_map[key]:
            f = resolve_factory(dotted)
            if f: candidates.append(f)
    for factory in candidates:
        try:
            return factory()
        except Exception:
            continue
    return LKBoxTracker()

class FaceTrack:
    __slots__ = ("id","tracker","bbox","label","conf","last_emb","last_refresh_idx","missed","age")
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
        self.age = 0  # frames alive

    def update(self, frame):
        ok, box = self.tracker.update(frame)
        self.age += 1
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

    def _assign_one_to_one(self, tracks_ids, tracks_boxes, det_boxes):
        """Greedy 1-1 assignment by IoU."""
        assignments = {}     # det_idx -> track_id
        used_tracks = set()
        used_dets = set()
        if not det_boxes or not tracks_ids:
            return assignments, set(range(len(det_boxes))), set(tracks_ids)

        # compute pairwise IoUs
        for _ in range(min(len(det_boxes), len(tracks_ids))):
            best_iou, best_pair = 0.0, None
            for di, db in enumerate(det_boxes):
                if di in used_dets: continue
                for ti, tid in enumerate(tracks_ids):
                    if tid in used_tracks: continue
                    iou = _iou(tuple(tracks_boxes[ti]), tuple(db))
                    if iou > best_iou:
                        best_iou = iou
                        best_pair = (di, tid, ti)
            if best_pair is None or best_iou < NMS_IOU_MATCH:
                break
            di, tid, ti = best_pair
            assignments[di] = tid
            used_dets.add(di)
            used_tracks.add(tid)

        unmatched_dets = set(range(len(det_boxes))) - used_dets
        unmatched_tracks = set(tracks_ids) - used_tracks
        return assignments, unmatched_dets, unmatched_tracks

    def _dedup_overlapping_tracks(self):
        """Merge/remove tracks that overlap heavily (visual duplicates)."""
        tids = list(self._tracks.keys())
        to_remove = set()
        for i in range(len(tids)):
            if tids[i] in to_remove: continue
            ti = self._tracks[tids[i]]
            for j in range(i+1, len(tids)):
                if tids[j] in to_remove: continue
                tj = self._tracks[tids[j]]
                iou = _iou(ti.bbox, tj.bbox)
                if iou >= TRACK_DEDUP_IOU:
                    # choose survivor: prefer labeled, higher conf, then older age
                    score_i = (ti.label != "Unknown", ti.conf, ti.age)
                    score_j = (tj.label != "Unknown", tj.conf, tj.age)
                    if score_i >= score_j:
                        to_remove.add(tids[j])
                    else:
                        to_remove.add(tids[i])
                        break
        for tid in to_remove:
            self._tracks.pop(tid, None)

    def match_and_update(self, frame, detected_boxes):
        """Update trackers, match detections 1-1, create new tracks, dedupe."""
        # 1) update/predict
        active_ids, active_boxes = [], []
        for tid, tr in list(self._tracks.items()):
            if tr.update(frame):
                active_ids.append(tid)
                active_boxes.append(tr.bbox)
            else:
                if tr.missed > TRACK_MAX_MISSED:
                    del self._tracks[tid]

        # 2) one-to-one assign
        assignments, unmatched_dets, _ = self._assign_one_to_one(active_ids, active_boxes, detected_boxes)

        # 3) re-init matched tracks on fresh boxes
        for di, tid in assignments.items():
            tr = self._tracks[tid]
            x1, y1, x2, y2 = map(int, detected_boxes[di])
            tr.tracker = _create_tracker()
            tr.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            tr.bbox = (x1, y1, x2, y2)

        # 4) new tracks for unmatched detections
        for di in unmatched_dets:
            x1, y1, x2, y2 = map(int, detected_boxes[di])
            tid = self._next_id; self._next_id += 1
            self._tracks[tid] = FaceTrack(tid, frame, (x1, y1, x2, y2))

        # 5) dedupe tracks that overlap a lot
        self._dedup_overlapping_tracks()
        return list(self._tracks.values())

    def update_only(self, frame):
        for tid, tr in list(self._tracks.items()):
            if not tr.update(frame) and tr.missed > TRACK_MAX_MISSED:
                del self._tracks[tid]
        self._dedup_overlapping_tracks()
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
    global _session_id, _session_started_ts, _session_dir
    global _accum, _csv_fh, _csv_writer, _pred_csv_fh, _pred_csv_writer
    global _perf_csv_fh, _perf_csv_writer, _yolo_csv_fh, _yolo_csv_writer
    global _track_mgr, _prev_yolo, _perf_sums, _conf_recognized, _conf_unknown, _yolo_class_totals

    _session_id = uuid.uuid4().hex[:8]
    _session_started_ts = time.time()

    _accum = {"frames": 0, "faces_total": 0, "recognized_total": 0, "unknown_total": 0,
              "avg_conf_overall": 0.0, "per_worker": {}}
    _perf_sums = {"arcface": 0.0, "yolo": 0.0, "total": 0.0}
    _conf_recognized = []; _conf_unknown = []; _yolo_class_totals = {}

    _track_mgr = FaceTrackManager()
    _prev_yolo = {"frame_idx": -10**9, "detections": None}

    ses_root = _ensure_dirs(log_dir or LOG_BASE_DIR)
    _session_dir = os.path.join(ses_root, _session_id)
    os.makedirs(_session_dir, exist_ok=True)

    # legacy CSV
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

    # predictions CSV
    pred_path = os.path.join(_session_dir, f"predictions_{_session_id}.csv")
    if _pred_csv_fh:
        try: _pred_csv_fh.close()
        except Exception: pass
    _pred_csv_fh = open(pred_path, "w", newline="")
    _pred_csv_writer = csv.writer(_pred_csv_fh)
    header = ["ts","session_id","frame_idx","track_id","x1","y1","x2","y2",
              "predicted","confidence","threshold","decision"]
    for k in range(1, TOPK_TO_LOG+1):
        header += [f"top{k}_name", f"top{k}_sim"]
    _pred_csv_writer.writerow(header)

    # perf CSV
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

    # YOLO CSV
    yolo_path = os.path.join(_session_dir, f"yolo_{_session_id}.csv")
    if _yolo_csv_fh:
        try: _yolo_csv_fh.close()
        except Exception: pass
    _yolo_csv_fh = open(yolo_path, "w", newline="")
    _yolo_csv_writer = csv.writer(_yolo_csv_fh)
    _yolo_csv_writer.writerow(["frame_idx","class","x1","y1","x2","y2"])

    # run-config snapshot
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
            "FACE_DET_MIN_SCORE": FACE_DET_MIN_SCORE,
            "FACE_DET_NMS_IOU": FACE_DET_NMS_IOU,
            "TRACK_DEDUP_IOU": TRACK_DEDUP_IOU,
            "TRACK_MAX_MISSED": TRACK_MAX_MISSED,
            "UNKNOWN_SUPPRESS_IOU": UNKNOWN_SUPPRESS_IOU,
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
    mat = np.stack([db[n] for n in names]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat = mat / norms
    return db, names, mat

# ------------------------
# Recognition helpers
# ------------------------
def _topk_embedding(emb_normed, names_arr, mat, k=3):
    sims = mat @ emb_normed  # (M,)
    if sims.size == 0: return []
    k = int(min(k, sims.shape[0]))
    idxs = np.argpartition(-sims, range(k))[:k]
    idxs = idxs[np.argsort(-sims[idxs])]
    return [(str(names_arr[i]), float(sims[i])) for i in idxs]

def _match_embedding(emb_normed, names_arr, mat):
    sims = mat @ emb_normed  # (M,)
    idx = int(np.argmax(sims))
    return names_arr[idx], float(sims[idx])
































# ------------------------
# Identity-level dedupe (ONE box per known worker)
# ------------------------
# --- One-box-per-worker controller ---
OWNER_TTL_FRAMES = 90              # ~3s at 30 FPS; adjust if your FPS is lower
OWNER_MISSING_GRACE_FRAMES = 12    # if owner not seen this many frames, allow switch
SWITCH_CONF_MARGIN = 0.06          # new track must beat owner by this conf margin to steal ownership

# PPE overlay: attach YOLO detections to the visible face track, don't draw extra rectangles
PPE_OVERLAY_IOU = 0.05             # min IoU to attach a YOLO detection to a face track

# Global: label -> {track_id, last_seen, conf}
_label_owner = {}

def _track_score(tr):
    # higher conf first, then older track, then bigger box
    return (float(tr.conf), tr.age, _area(tr.bbox))

def _enforce_single_owner(tracks, frame_idx):
    """
    For every label != 'Unknown', keep ONE canonical owner track across frames.
    All non-owner tracks with the same label are force-suppressed.
    """
    global _label_owner

    # Group current tracks by label (ignore Unknown/Suppressed)
    current = {}
    for tr in tracks:
        if tr.label in ("Unknown", "Suppressed"):
            continue
        current.setdefault(tr.label, []).append(tr)

    # Update/choose owner for each label seen this frame
    for label, cand in current.items():
        # Best candidate this frame
        best = max(cand, key=_track_score)

        owner = _label_owner.get(label)
        if owner is None:
            # no owner yet -> pick best now
            _label_owner[label] = {
                "track_id": best.id,
                "last_seen": frame_idx,
                "conf": float(best.conf)
            }
        else:
            # owner exists
            owner_tr = next((t for t in cand if t.id == owner["track_id"]), None)

            # if owner not present recently -> allow takeover
            if owner_tr is None:
                stale = (frame_idx - owner["last_seen"]) > OWNER_MISSING_GRACE_FRAMES
                if stale:
                    _label_owner[label] = {
                        "track_id": best.id,
                        "last_seen": frame_idx,
                        "conf": float(best.conf)
                    }
                # else keep old owner until grace expires
            else:
                # owner is present: update last_seen/conf
                owner["last_seen"] = frame_idx
                owner["conf"] = float(owner_tr.conf)

                # allow switching only if challenger beats owner by margin
                if best.id != owner_tr.id and float(best.conf) >= float(owner_tr.conf) + SWITCH_CONF_MARGIN:
                    _label_owner[label] = {
                        "track_id": best.id,
                        "last_seen": frame_idx,
                        "conf": float(best.conf)
                    }

    # Expire old owners that haven't been seen for too long
    to_del = []
    for label, owner in _label_owner.items():
        if (frame_idx - owner["last_seen"]) > OWNER_TTL_FRAMES:
            to_del.append(label)
    for label in to_del:
        del _label_owner[label]

    # Finally, suppress all non-owner duplicates this frame
    for tr in tracks:
        if tr.label in ("Unknown", "Suppressed"):
            continue
        owner = _label_owner.get(tr.label)
        if not owner:
            continue  # no owner registered yet (rare): let best become owner next frame
        if tr.id != owner["track_id"]:
            tr.label = "Suppressed"
            tr.conf = -1.0


def _select_visible_tracks(tracks):
    """
    After owner enforcement:
      - Keep exactly one track per known label (the owner/best).
      - Keep ALL 'Unknown'.
      - Drop any 'Suppressed'.
    """
    best_by_label = {}
    unknown_tracks = []

    for tr in tracks:
        if tr.label == "Suppressed":
            continue
        if tr.label == "Unknown":
            unknown_tracks.append(tr)
        else:
            cur = best_by_label.get(tr.label)
            if cur is None or _track_score(tr) > _track_score(cur):
                best_by_label[tr.label] = tr

    visible_tracks = list(best_by_label.values()) + unknown_tracks
    visible_ids = {tr.id for tr in visible_tracks}
    return visible_tracks, visible_ids


def _attach_ppe_to_tracks(yolo_boxes, tracks, iou_thr=PPE_OVERLAY_IOU):
    """
    Link YOLO PPE detections to the SINGLE visible face track they overlap best.
    Returns: dict track_id -> list[str] of class names (unique).
    yolo_boxes: [(x1,y1,x2,y2,class_name), ...]
    """
    if not tracks or not yolo_boxes:
        return {}

    ppe = {tr.id: [] for tr in tracks}
    for (x1, y1, x2, y2, cls_name) in yolo_boxes:
        best_tr, best_iou = None, 0.0
        for tr in tracks:
            iou = _iou(tr.bbox, (x1, y1, x2, y2))
            if iou > best_iou:
                best_iou, best_tr = iou, tr
        if best_tr is not None and best_iou >= iou_thr:
            lst = ppe.setdefault(best_tr.id, [])
            if cls_name not in lst:
                lst.append(cls_name)
    return ppe


# --- HARD DELETE helper: remove non-owner duplicate tracks physically ---
def _hard_delete_duplicate_labels(frame_idx):
    """
    Physically remove all non-owner tracks that share the same known label.
    Also removes any tracks currently marked 'Suppressed'.
    """
    # snapshot of current tracks
    all_tracks = list(_track_mgr._tracks.values())

    # 1) remove all 'Suppressed' immediately
    suppressed_ids = [tr.id for tr in all_tracks if tr.label == "Suppressed"]
    for tid in suppressed_ids:
        _track_mgr._tracks.pop(tid, None)

    # refresh snapshot
    all_tracks = list(_track_mgr._tracks.values())
    if not all_tracks:
        return

    # 2) for each known label, keep only the owner (or best if owner missing)
    #    and delete the rest
    # build groups by label
    grouped = {}
    for tr in all_tracks:
        if tr.label in ("Unknown", "Suppressed"):
            continue
        grouped.setdefault(tr.label, []).append(tr)

    for label, cand in grouped.items():
        # choose keep id: prefer registered owner id if present, else best-scored
        owner = _label_owner.get(label)
        keep_id = owner["track_id"] if (owner and any(t.id == owner["track_id"] for t in cand)) else None
        if keep_id is None:
            best = max(cand, key=_track_score)
            keep_id = best.id
            # refresh owner with the best we keep
            _label_owner[label] = {"track_id": keep_id, "last_seen": frame_idx, "conf": float(best.conf)}
        # delete non-keepers
        for tr in cand:
            if tr.id != keep_id:
                _track_mgr._tracks.pop(tr.id, None)


# ------------------------
# Recognition on Frames (with tracking & duty-cycling)
# ------------------------
def process_frame(frame, db_pack, threshold=0.35, frame_idx=0):
    db_dict, db_names, db_mat = db_pack
    t0 = time.time()
    stats = {
        "faces": 0, "recognized": 0, "unknown": 0,
        "recognized_workers": [], "yolo_detections": {}, "timing_ms": {}
    }

    raw = frame
    display = frame.copy()

    global _accum, _csv_writer, _pred_csv_writer, _prev_yolo
    global _perf_sums, _conf_recognized, _conf_unknown, _yolo_csv_writer, _yolo_class_totals

    # --------- Face detect vs track ----------
    detect_frame = (frame_idx % DETECT_EVERY_N == 0)
    faces = []
    if detect_frame:
        faces_all = arcface_model.get(raw)  # faces have .bbox, .det_score, .embedding
        det_boxes, det_scores = [], []
        keep_faces = []
        for f in faces_all:
            score = float(getattr(f, "det_score", 1.0))
            x1, y1, x2, y2 = map(int, f.bbox)
            if score < FACE_DET_MIN_SCORE:
                continue
            if min(x2-x1, y2-y1) < MIN_FACE_SIZE:
                continue
            det_boxes.append((x1, y1, x2, y2))
            det_scores.append(score)
            keep_faces.append(f)

        # NMS among face detections
        if det_boxes:
            keep_idx = _nms_indices(det_boxes, det_scores, FACE_DET_NMS_IOU)
            faces = [keep_faces[i] for i in keep_idx]
            det_boxes = [det_boxes[i] for i in keep_idx]
        else:
            faces = []
            det_boxes = []

        tracks = _track_mgr.match_and_update(raw, det_boxes)
    else:
        tracks = _track_mgr.update_only(raw)

    # Compute/refresh identity for each track
    for tr in tracks:
        x1, y1, x2, y2 = tr.bbox
        need_refresh = detect_frame or (frame_idx - tr.last_refresh_idx >= EMBED_EVERY_M)
        if need_refresh:
            x1c = max(0, x1); y1c = max(0, y1); x2c = min(display.shape[1]-1, x2); y2c = min(display.shape[0]-1, y2)
            crop = raw[y1c:y2c, x1c:x2c]
            if crop.size > 0:
                emb = None
                if detect_frame and faces:
                    # reuse full-frame face embedding by best IoU
                    best_i, best_iou = -1, 0.0
                    for i, f in enumerate(faces):
                        fb = tuple(map(int, f.bbox))
                        iou = _iou(fb, tr.bbox)
                        if iou > best_iou:
                            best_i, best_iou = i, iou
                    if best_i >= 0:
                        emb = faces[best_i].embedding.astype(np.float32)
                if emb is None:
                    f2 = arcface_model.get(crop)
                    if f2:
                        emb = f2[0].embedding.astype(np.float32)

                if emb is not None:
                    emb /= (np.linalg.norm(emb) + 1e-9)
                    tr.last_emb = emb
                    tr.last_refresh_idx = frame_idx

                    if db_mat is not None and db_names is not None and db_mat.size > 0:
                        topk = _topk_embedding(emb, db_names, db_mat, TOPK_TO_LOG)
                        best_match, best_score = topk[0] if len(topk) else (None, -1.0)
                    else:
                        topk = []
                        best_match, best_score = None, -1.0

                    if best_match is not None and best_score >= threshold:
                        tr.label = best_match
                        tr.conf  = best_score
                    else:
                        tr.label = "Unknown"
                        tr.conf  = best_score if best_score is not None else -1.0

    # mark end of ArcFace/ID part for timing
    t_arc_end = time.time()

    # --- Enforce single owner per known worker label (kills duplicate boxes visually) ---
    _enforce_single_owner(tracks, frame_idx)

    # --- HARD PRUNE: delete non-owner duplicates so ghosts cannot persist ---
    # Do it on every detect frame (when identities refresh) and also every ~3s.
    if detect_frame or (frame_idx % OWNER_TTL_FRAMES == 0):
        _hard_delete_duplicate_labels(frame_idx)

    # --- Decide which tracks to render (one per known label + all unknowns) ---
    visible_tracks, visible_ids = _select_visible_tracks(tracks)

    # ---------- YOLO detection (duty-cycled, keep stats, no rectangles) ----------
    if frame_idx - _prev_yolo["frame_idx"] >= YOLO_EVERY_N:
        det = yolo_model(raw)[0]
        yolo_stats = {}
        yolo_boxes = []
        for box, cls_id in zip(det.boxes.xyxy, det.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_name = det.names[int(cls_id)]
            yolo_stats[class_name] = yolo_stats.get(class_name, 0) + 1
            yolo_boxes.append((x1, y1, x2, y2, class_name))
            if _yolo_csv_writer:
                _yolo_csv_writer.writerow([frame_idx, class_name, x1, y1, x2, y2])
        _prev_yolo = {"frame_idx": frame_idx, "detections": (yolo_stats, yolo_boxes)}
        for k, v in yolo_stats.items():
            _yolo_class_totals[k] = _yolo_class_totals.get(k, 0) + v
    else:
        yolo_stats, yolo_boxes = _prev_yolo["detections"] if _prev_yolo["detections"] else ({}, [])

    stats["yolo_detections"] = yolo_stats

    # Attach PPE detections to *visible* face tracks (NO extra boxes)
    ppe_map = _attach_ppe_to_tracks(yolo_boxes, visible_tracks, iou_thr=PPE_OVERLAY_IOU)

    # ------- draw & log only visible tracks (single box per known worker) -------
    known_now = 0
    stats["faces"] = len(visible_tracks)  # what we actually render
    now = time.time()

    # for UI: unique list of recognized worker names in the frame
    stats["recognized_workers"] = sorted({tr.label for tr in visible_tracks if tr.label != "Unknown"})

    for tr in visible_tracks:
        x1, y1, x2, y2 = tr.bbox
        ppe_tags = ppe_map.get(tr.id, [])
        ppe_suffix = (" | " + ",".join(ppe_tags)) if ppe_tags else ""

        if tr.label != "Unknown":
            known_now += 1
            _accum["recognized_total"] += 1
            prev_n = _accum["recognized_total"] - 1
            _accum["avg_conf_overall"] = _rolling_avg(_accum["avg_conf_overall"], prev_n, tr.conf)
            w = _accum["per_worker"].setdefault(tr.label, {"seen": 0, "recognized": 0, "avg_conf": 0.0})
            w["seen"] += 1; w["recognized"] += 1
            w["avg_conf"] = _rolling_avg(w["avg_conf"], w["recognized"] - 1, tr.conf)
            _conf_recognized.append(tr.conf)

            color = (0, 255, 0)
            label = f"{tr.label} ({tr.conf:.2f}){ppe_suffix}"
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if _csv_writer:
                _csv_writer.writerow([now, _session_id, frame_idx, len(visible_tracks), 1, 0,
                                      tr.label, round(tr.conf, 4), x1, y1, x2, y2])
            if _pred_csv_writer:
                _pred_csv_writer.writerow([now, _session_id, frame_idx, tr.id, x1, y1, x2, y2,
                                           tr.label, round(tr.conf, 6), threshold, "recognized"])
        else:
            _accum["unknown_total"] += 1
            _conf_unknown.append(tr.conf)

            color = (0, 0, 255)
            label = "Unknown" if tr.conf is None or tr.conf < 0 else f"Unknown ({tr.conf:.2f})"
            if ppe_suffix:
                label += ppe_suffix
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if _csv_writer:
                _csv_writer.writerow([now, _session_id, frame_idx, len(visible_tracks), 0, 1,
                                      "Unknown", round(tr.conf if tr.conf else -1.0, 4), x1, y1, x2, y2])
            if _pred_csv_writer:
                _pred_csv_writer.writerow([now, _session_id, frame_idx, tr.id, x1, y1, x2, y2,
                                           "Unknown", round(tr.conf if tr.conf else -1.0, 6), threshold, "unknown"])

    # mark end of YOLO/overlay part for timing
    t_end = time.time()

    # instant counts for UI (based on visible set)
    stats["recognized"] = known_now
    stats["unknown"] = max(0, len(visible_tracks) - known_now)

    # timing breakdown
    arc_ms = (t_arc_end - t0) * 1000.0
    yolo_ms = (t_end - t_arc_end) * 1000.0
    tot_ms  = (t_end - t0) * 1000.0
    fps = 1000.0 / max(1e-6, tot_ms)
    stats["timing_ms"] = {"arcface": round(arc_ms, 2), "yolo": round(yolo_ms, 2), "total": round(tot_ms, 2)}

    if _perf_csv_writer:
        _perf_csv_writer.writerow([
            frame_idx, round(arc_ms,2), round(yolo_ms,2), round(tot_ms,2), round(fps,2),
            len(visible_tracks), known_now, max(0, len(visible_tracks)-known_now)
        ])

    _accum["frames"] += 1
    _accum["faces_total"] += len(visible_tracks)  # count what's shown
    _perf_sums["arcface"] += arc_ms
    _perf_sums["yolo"] += yolo_ms
    _perf_sums["total"] += tot_ms

    return display, stats






















# -------- Summary / Accuracy --------
def _maybe_load_ground_truth(gt_path):
    if not os.path.exists(gt_path): return None
    gt = {}
    with open(gt_path, "r", newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            try:
                fi = int(row["frame_idx"]); tid = int(row["track_id"]); lbl = str(row["true_worker"]).strip()
                gt[(fi, tid)] = lbl
            except Exception:
                continue
    return gt

def _compute_accuracy_from_gt(pred_csv_path, gt_map):
    if gt_map is None: return None
    classes = set(gt_map.values())
    TP = {c:0 for c in classes}; FP = {c:0 for c in classes}; FN = {c:0 for c in classes}
    conf_matrix = {}
    matched = 0
    with open(pred_csv_path, "r", newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            fi = int(row["frame_idx"]); tid = int(row["track_id"])
            decision = row["decision"]; pred = row["predicted"].strip()
            if (fi, tid) not in gt_map: continue
            matched += 1
            true = gt_map[(fi, tid)]
            if decision == "recognized":
                if pred == true:
                    TP[true] = TP.get(true,0) + 1
                else:
                    FP[pred] = FP.get(pred,0) + 1
                    FN[true] = FN.get(true,0) + 1
                conf_matrix.setdefault(true, {})
                conf_matrix[true][pred] = conf_matrix[true].get(pred, 0) + 1
            else:
                FN[true] = FN.get(true,0) + 1
                conf_matrix.setdefault(true, {})
                conf_matrix[true]["Unknown"] = conf_matrix[true].get("Unknown", 0) + 1

    per_class = {}
    for c in sorted(set(list(TP.keys()) + list(FP.keys()) + list(FN.keys()))):
        tp, fp, fn = TP.get(c,0), FP.get(c,0), FN.get(c,0)
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = (2*prec*rec) / max(1e-9, (prec+rec))
        per_class[c] = {"precision": round(prec,4), "recall": round(rec,4), "f1": round(f1,4), "tp": tp, "fp": fp, "fn": fn}

    total_tp = sum(TP.values()); total = matched
    acc = total_tp / max(1, total)
    micro_prec = total_tp / max(1, total_tp + sum(FP.values()))
    micro_rec  = total_tp / max(1, total_tp + sum(FN.values()))
    micro_f1   = (2*micro_prec*micro_rec) / max(1e-9, (micro_prec+micro_rec))
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
            "FACE_DET_MIN_SCORE": FACE_DET_MIN_SCORE,
            "FACE_DET_NMS_IOU": FACE_DET_NMS_IOU,
            "TRACK_DEDUP_IOU": TRACK_DEDUP_IOU,
            "TRACK_MAX_MISSED": TRACK_MAX_MISSED,
            "UNKNOWN_SUPPRESS_IOU": UNKNOWN_SUPPRESS_IOU,
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

    gt_path = os.path.join(_session_dir, "ground_truth.csv")
    acc = _maybe_load_ground_truth(gt_path)
    if acc is not None:
        preds_path = summary["files"]["predictions_csv"]
        metrics = _compute_accuracy_from_gt(preds_path, acc)
        summary["ground_truth_metrics"] = metrics
        summary["files"]["ground_truth_csv"] = gt_path

    with open(os.path.join(_session_dir, f"summary_{_session_id}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

# ------------------------
# Streaming Generator
# ------------------------
def recognize_from_video_stream(video_path=None, threshold=0.35, log_dir=LOG_BASE_DIR):
    """
    If video_path is None or 'camera' (case-insensitive), open the default webcam.
    On macOS, use AVFoundation for reliability.
    """
    global latest_stats
    csv_path = _reset_session(log_dir=log_dir)
    db_pack = load_worker_db(DB_PATH)

    # ---- camera vs file selection (robust for macOS) ----
    use_camera = (video_path is None) or (str(video_path).strip().lower() == "camera")
    if use_camera:
        if sys.platform == "darwin":
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

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

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()
        _finalize_session(db_pack, threshold)
        try:
            if _csv_fh: _csv_fh.close()
            if _pred_csv_fh: _pred_csv_fh.close()
            if _perf_csv_fh: _perf_csv_fh.close()
            if _yolo_csv_fh: _yolo_csv_fh.close()
        except Exception:
            pass
