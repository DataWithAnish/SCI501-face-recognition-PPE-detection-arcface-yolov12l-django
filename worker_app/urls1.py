from django.urls import path
from . import views1

urlpatterns = [
    path("video/", views1.video_page, name="video_page"),
    path("video_feed/", views1.video_feed, name="video_feed"),
    path("workers/", views1.worker_list, name="worker_list"),
    path("workers/upload/", views1.upload_worker, name="upload_worker"),
    path("video_stats/", views1.video_stats, name="video_stats"),

]
