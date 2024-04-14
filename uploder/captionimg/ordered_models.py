from django.db import models

class OrderedIDModel(models.Model):
    ordered_id = models.PositiveIntegerField(unique=True)

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        if not self.ordered_id:
            max_id = self.__class__.objects.aggregate(models.Max('ordered_id'))['ordered_id__max']
            self.ordered_id = 1 if max_id is None else max_id + 1
        super().save(*args, **kwargs)
