from django.db import models


class ReviewLevel(models.TextChoices):
    # Gated 2-reviews
    RED = ("red", "Red Level")
    # Gated single review
    YELLOW = ("yellow", "Yello Level")
    # Non gated single review
    GREEN = ("green", "Green Level")
    # Non gated spot checks
    GOLD = ("gold", "Gold Level")
