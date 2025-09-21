from django.db import models
import os
class Worker(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name
def worker_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/workers/<worker_name>/<filename>
    return os.path.join("workers", instance.worker.name, filename)
class WorkerImage(models.Model):
    worker = models.ForeignKey(Worker, on_delete=models.CASCADE, related_name="images")
    image = models.ImageField(upload_to=worker_directory_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
