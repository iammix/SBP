from django.db import models

# Create your models here.
class TodoItem(models.Model):
    titile = models.CharField(max_length=200)
    completed = models.BooleanField(default=False)