from django.contrib import admin
from django.db import models

from core.portal.lib.models.base import Model
from core.portal.perceived_data.models.perceived_event_rate_definition import (
    PerceivedEventRateDefinition,
)
from core.portal.scores.models.score_band import ScoreBand


class ScoreDefinition(Model):
    class Meta:
        app_label = "scores"
        db_table = "score_definition"
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "perceived_event_rate_definition",
                    "score_band",
                    "calculation_method",
                ],
                name="score_definition_unique_constraint",
            )
        ]

    name = models.CharField(
        max_length=100,
    )

    perceived_event_rate_definition = models.ForeignKey(
        PerceivedEventRateDefinition,
        on_delete=models.CASCADE,
        null=True,
    )

    score_band = models.ForeignKey(
        ScoreBand,
        on_delete=models.PROTECT,
    )

    class CalculationMethod(models.IntegerChoices):
        NOT_IMPLEMENTED = 0
        THIRTY_DAY_EVENT_SCORE = 1

    calculation_method = models.PositiveSmallIntegerField(
        blank=False,
        null=False,
        default=CalculationMethod.NOT_IMPLEMENTED,
        choices=CalculationMethod.choices,
    )

    def __str__(self) -> str:
        """Returns string representation of class

        Returns:
            str: string representation of class
        """
        return (
            f"{self.name} : {self.perceived_event_rate_definition}"
            f"- {self.score_band} - {self.calculation_method}"
            f"- (pk: {self.pk})"
        )


admin.site.register(ScoreDefinition)
