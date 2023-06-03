from django.contrib import admin
from django.db import models

from core.portal.devices.models.camera import Camera, Zone
from core.portal.lib.models.base import Model
from core.portal.scores.models.score_definition import ScoreDefinition


class SiteScoreConfig(Model):
    class Meta:
        app_label = "scores"
        db_table = "site_score_config"
        # TODO: check constraint that camera belongs to Site if camera exists
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "site",
                    "camera",
                    "score_definition",
                ],
                name="site_score_config_unique_constraint",
            )
        ]

    name = models.CharField(
        unique=True,
        max_length=100,
    )

    site = models.ForeignKey(
        Zone,
        on_delete=models.CASCADE,
    )

    camera = models.ForeignKey(
        Camera,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )

    score_definition = models.ForeignKey(
        ScoreDefinition,
        on_delete=models.CASCADE,
    )

    def __str__(self) -> str:
        """Returns string representation of class

        Returns:
            str: string representation of class
        """
        return (
            f"{self.name}: {self.site} - {self.camera} - {self.score_definition}"
            f"- (pk: {self.pk})"
        )

    # TODO: visible, enabled, metadata (title, band colors, ui options)


admin.site.register(SiteScoreConfig)
