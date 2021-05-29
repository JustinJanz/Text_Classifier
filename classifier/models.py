from django.db import models

# Create your models here.
class Text(models.Model):
    text = models.TextField()
    category = models.CharField(max_length=100)
    sub_category = models.CharField(max_length=100)

    def __str__(self):
        return self.category
