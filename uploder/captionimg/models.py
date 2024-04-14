from django.db import models

class Img_model(models.Model):
    img = models.ImageField(upload_to='images/')