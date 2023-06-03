from django.db import models


class TimeBucketWidth(models.IntegerChoices):
    """Time bucket width choices."""

    YEAR = 1
    QUARTER = 2
    MONTH = 3
    WEEK = 4
    DAY = 5
    HOUR = 6
