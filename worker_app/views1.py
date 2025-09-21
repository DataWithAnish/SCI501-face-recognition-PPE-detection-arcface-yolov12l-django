import os
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
from django.conf import settings
from .models import Worker, WorkerImage
from .forms import WorkerUploadForm
from recognition.arcface_yolo2 import recognize_from_video_stream, build_worker_db

# --------------------
# Video Streaming Views
# --------------------
VIDEO_PATH = "/Users/anishkumarshrestha/Desktop/PIE detector/SCI501/output_worker1.mp4"

def video_feed(request):
    """Stream video frames from file as MJPEG"""
    return StreamingHttpResponse(
        recognize_from_video_stream(video_path=VIDEO_PATH),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )

def video_page(request):
    """Render a page with video stream"""
    return render(request, "workers/video.html")

# --------------------
# Worker Management Views
# --------------------
def worker_list(request):
    """Show all workers and their image counts"""
    workers = Worker.objects.all()
    return render(request, "workers/list.html", {"workers": workers})

def upload_worker(request):
    """Upload new worker or add images to existing one"""
    if request.method == "POST":
        print("🟢 DEBUG: POST request received")

        print("🟢 DEBUG: request.FILES ->", request.FILES)
        print("🟢 DEBUG: keys in request.FILES ->", list(request.FILES.keys()))

        form = WorkerUploadForm(request.POST, request.FILES)
        print("🟢 DEBUG: form.is_valid() ->", form.is_valid())
        print("🟢 DEBUG: form.errors ->", form.errors)

        if form.is_valid():
            name = form.cleaned_data["name"]
            images = request.FILES.getlist("images")
            print("🟢 DEBUG: Uploaded files list ->", images)

            # check if worker exists
            worker, created = Worker.objects.get_or_create(name=name)

            # save uploaded images
            for img in images:
                print(f"🟢 DEBUG: Saving {img.name} for worker {name}")
                WorkerImage.objects.create(worker=worker, image=img)

            # rebuild embeddings DB
            build_worker_db(
                workers_dir=os.path.join(settings.MEDIA_ROOT, "workers"),
                save_path=os.path.join(os.path.dirname(__file__), "../recognition/workers_db.npy")
            )

            return redirect("worker_list")
    else:
        print("🟡 DEBUG: GET request (form load)")
        form = WorkerUploadForm()

    return render(request, "workers/upload.html", {"form": form})
from django.http import JsonResponse
from recognition.arcface_yolo2 import latest_stats

def video_stats(request):
    from recognition.arcface_yolo2 import latest_stats
    print("DEBUG: API returned stats ->", latest_stats)  # 🟢 check in terminal
    return JsonResponse(latest_stats or {"status": "waiting for frames"})
