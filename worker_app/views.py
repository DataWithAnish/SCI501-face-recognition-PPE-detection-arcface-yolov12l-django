# worker_app/views.py
import importlib
from pathlib import Path

from django.conf import settings
from django.http import (
    StreamingHttpResponse,
    JsonResponse,
    Http404,
    HttpResponseServerError,
    HttpResponseRedirect,
)
from django.shortcuts import render, redirect
from django.urls import reverse

from .models import Worker, WorkerImage
from .forms import WorkerUploadForm

# --------------------
# Load the pipeline module SAFELY
# --------------------
# If you changed the filename, adjust the string below.
_recog = importlib.import_module("recognition.arcface_yolo")

# Pull required symbols
recognize_from_video_stream = getattr(_recog, "recognize_from_video_stream")
build_worker_db            = getattr(_recog, "build_worker_db")
latest_stats               = getattr(_recog, "latest_stats")

# Fallback logger: prefer pipeline’s `_debug`, else use a flushed print
_debug = getattr(_recog, "_debug", lambda msg: print(msg, flush=True))

# --------------------
# Config
# --------------------
DEFAULT_VIDEO_PATH = Path(settings.BASE_DIR) / "output_worker1.mp4"
RECOG_LOG_DIR = Path(settings.BASE_DIR) / "recognition"


# --------------------
# Helpers
# --------------------
def _coerce_float(val: str | None, default: float) -> float:
    if not val:
        return default
    try:
        return float(val)
    except Exception:
        return default


def _streaming_response(gen) -> StreamingHttpResponse:
    """
    Wrap the generator with StreamingHttpResponse and add headers that help
    browsers/proxies not buffer the multipart stream too aggressively.
    """
    resp = StreamingHttpResponse(
        gen, content_type="multipart/x-mixed-replace; boundary=frame"
    )
    resp["X-Accel-Buffering"] = "no"  # disable buffering (nginx)
    resp["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp["Pragma"] = "no-cache"
    return resp


def _resolve_source(source: str | None) -> str:
    """
    Normalize source:
      - None -> DEFAULT_VIDEO_PATH if exists, else 'camera'
      - 'camera' -> 'camera'
      - file path -> absolute string path (404 if not found)
    """
    if not source:
        if DEFAULT_VIDEO_PATH.exists():
            return str(DEFAULT_VIDEO_PATH)
        return "camera"

    source = source.strip()
    if source.lower() == "camera":
        return "camera"

    p = Path(source).expanduser().resolve()
    if not p.exists():
        raise Http404(f"Video file not found: {p}")
    return str(p)


# --------------------
# Root -> Video page
# --------------------
def home(request):
    """Redirect '/' to the simple video page."""
    return HttpResponseRedirect(reverse("video_page"))


# --------------------
# Video Streaming Views
# --------------------
def video_feed(request):
    """
    Stream video (camera or file) as MJPEG.

    Examples:
      /video_feed/?source=camera
      /video_feed/?source=/full/path/to/file.mp4
      /video_feed/?threshold=0.35
    """
    try:
        source_qs = (request.GET.get("source") or "").strip() or None
        threshold = _coerce_float(request.GET.get("threshold"), 0.35)

        source = _resolve_source(source_qs)

        # Log exactly which module/file is being used
        try:
            pipe_file = getattr(recognize_from_video_stream, "__code__", None).co_filename
        except Exception:
            pipe_file = str(getattr(_recog, "__file__", "<?>"))
        _debug(f"view.video_feed → pipeline file: {pipe_file}")

        if source == "camera":
            _debug(f"🎥 view.video_feed starting | source=camera | threshold={threshold}")
            video_path = None  # camera for the pipeline
        else:
            _debug(f"🎬 view.video_feed starting | file={source} | threshold={threshold}")
            video_path = source

        # Wrap the generator to log lifecycle
        def gen():
            _debug("▶️  generator: open stream")
            try:
                for chunk in recognize_from_video_stream(
                    video_path=video_path,
                    threshold=threshold,
                    log_dir=str(RECOG_LOG_DIR),
                ):
                    yield chunk
            except Exception as e:
                _debug(f"❌ generator error: {e!r}")
                raise
            finally:
                _debug("⏹️  generator: stream ended")

        return _streaming_response(gen())

    except Http404:
        raise
    except Exception as e:
        _debug(f"❌ Video feed error: {e!r}")
        return HttpResponseServerError(f"Video feed error: {e}")


def video_page(request):
    """
    Simple page with the <img> tag pointing to the MJPEG stream.
    """
    ctx = {
        "camera_url": f"{reverse('video_feed')}?source=camera",
        "file_url": (
            f"{reverse('video_feed')}?source={DEFAULT_VIDEO_PATH}"
            if DEFAULT_VIDEO_PATH.exists()
            else ""
        ),
        "stats_url": reverse("video_stats"),
    }
    return render(request, "workers/video.html", ctx)


def video_stats(request):
    """
    Returns the latest per-frame and cumulative stats the pipeline writes.
    Your template can poll this once per second.
    """
    return JsonResponse(latest_stats or {"status": "waiting for frames"}, safe=False)


# --------------------
# Worker Management Views
# --------------------
def worker_list(request):
    """Show all workers and their image counts"""
    workers = Worker.objects.all()
    return render(request, "workers/list.html", {"workers": workers})


def upload_worker(request):
    """
    Upload a new worker or add images to an existing one, then rebuild embeddings DB.
    """
    if request.method == "POST":
        _debug("🟢 DEBUG: POST /upload_worker received")
        form = WorkerUploadForm(request.POST, request.FILES)
        _debug(f"🟢 DEBUG: form.is_valid()={form.is_valid()} errors={form.errors}")

        if form.is_valid():
            name = form.cleaned_data["name"]
            images = request.FILES.getlist("images")
            _debug(f"🟢 DEBUG: saving {len(images)} images for worker '{name}'")

            worker, _ = Worker.objects.get_or_create(name=name)

            for img in images:
                WorkerImage.objects.create(worker=worker, image=img)

            workers_dir = Path(settings.MEDIA_ROOT) / "workers"
            save_path = Path(settings.BASE_DIR) / "recognition" / "workers_db.npy"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            _debug(f"🛠️  Rebuilding embeddings DB from: {workers_dir}")
            _debug(f"🛟  Saving workers_db.npy to: {save_path}")
            build_worker_db(workers_dir=str(workers_dir), save_path=str(save_path))

            return redirect("worker_list")
    else:
        _debug("🟡 DEBUG: GET /upload_worker (render form)")
        form = WorkerUploadForm()

    return render(request, "workers/upload.html", {"form": form})
