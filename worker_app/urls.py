# worker_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),                  # NEW: root -> video page
    path("video/", views.video_page, name="video_page"),
    path("video_feed/", views.video_feed, name="video_feed"),
    path("workers/", views.worker_list, name="worker_list"),
    path("workers/upload/", views.upload_worker, name="upload_worker"),
    path("video_stats/", views.video_stats, name="video_stats"),
]
