from django.contrib import admin
from django.db import models

from core.portal.lib.models.base import Model
from core.portal.scores.models.score_band import ScoreBand


class ScoreBandRange(Model):
    class Meta:
        app_label = "scores"
        db_table = "score_band_range"
        # TODO: add CHECK constraint on score_band:
        # For any two rows a and b: CHECK(a, b) == !CHECK(b, a) where a.score_band == b.score_band
        # CHECK(a, b): a.lower_bound < b.lower_bound && a.score_value > b.score_value
        # Add constraint for deleted_at.
        constraints = [
            models.UniqueConstraint(
                # TODO: do we need an additional unique Contrainst: (score_band, score_value)
                fields=[
                    "score_band",
                    "lower_bound_inclusive",
                    # "deleted_at",
                ],
                name="score_band_range_unique_constraint",
            )
        ]

    score_band = models.ForeignKey(
        ScoreBand,
        on_delete=models.CASCADE,
    )

    lower_bound_inclusive = models.DecimalField(
        blank=False,
        null=False,
        default=0.0,
        # This may be overkill, but this field supports the following range:
        #     min: 000000000.0000000001
        #     max: 999999999.9999999999
        max_digits=19,
        decimal_places=10,
        help_text="The lower bound of the range",
    )

    score_value = models.PositiveSmallIntegerField()

    def __str__(self) -> str:
        """Returns string representation of class

        Returns:
            str: string representation of class
        """
        return (
            f"{self.score_band} - ({self.lower_bound_inclusive}: {self.score_value})"
            f"- (pk: {self.pk})"
        )


admin.site.register(ScoreBandRange)
