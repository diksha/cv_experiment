from django.db import models


class AggregateGroup(models.TextChoices):
    """Aggregate group choices."""

    HOUR = "HOUR"
    DAY = "DAY"
    MONTH = "MONTH"
    YEAR = "YEAR"
