from django.db import models


class ZoneType(models.TextChoices):
    SITE = "site"
    ROOM = "room"
    AREA = "area"
