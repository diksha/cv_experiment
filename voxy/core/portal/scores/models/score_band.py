from django.contrib import admin
from django.db import models

from core.portal.lib.models.base import Model


class ScoreBand(Model):
    class Meta:
        app_label = "scores"
        db_table = "score_band"

    name = models.CharField(
        unique=True,
        max_length=100,
    )

    def __str__(self) -> str:
        """Returns string representation of class

        Returns:
            str: string representation of class
        """
        return f"{self.name} - {self.pk}"


admin.site.register(ScoreBand)
