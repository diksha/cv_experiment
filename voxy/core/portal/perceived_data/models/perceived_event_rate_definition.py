from django.contrib import admin
from django.db import models

from core.portal.api.models.incident_type import IncidentType
from core.portal.lib.models.base import Model
from core.portal.perceived_data.enums import (
    PerceivedActorStateDurationCategory,
    PerceivedEventRateCalculationMethod,
)


class PerceivedEventRateDefinition(Model):
    """Perceived event rate definition."""

    class Meta:
        """Django meta class used to configure the model."""

        app_label = "perceived_data"
        db_table = "perceived_event_rate_definition"

        constraints = [
            models.UniqueConstraint(
                fields=[
                    "incident_type",
                    "calculation_method",
                    "perceived_actor_state_duration_category",
                ],
                name="perceived_event_rate_definition_unique_constraint",
            ),
        ]

    incident_type = models.ForeignKey(
        IncidentType,
        on_delete=models.CASCADE,
        blank=False,
        null=False,
        related_name="perceived_event_rate_definitions",
    )
    name = models.CharField(
        blank=False,
        null=False,
        max_length=100,
    )
    calculation_method = models.PositiveSmallIntegerField(
        blank=False,
        null=False,
        choices=PerceivedEventRateCalculationMethod.choices,
    )
    perceived_actor_state_duration_category = models.PositiveSmallIntegerField(
        blank=False,
        null=False,
        choices=PerceivedActorStateDurationCategory.choices,
    )

    def __str__(self) -> str:
        """String representation of the instance.

        Returns:
            str: string representation of the instance.
        """

        return self.name


admin.site.register(PerceivedEventRateDefinition)
