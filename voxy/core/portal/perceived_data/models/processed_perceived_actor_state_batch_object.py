from django.db import models

from core.portal.lib.models.base import Model


class ProcessedPerceivedActorStateBatchObject(Model):
    """The S3 objects that have already been processed"""

    class Meta:
        """Django meta class used to configure the model."""

    app_label = "perceived_data"
    db_table = "processed_perceived_actor_state_batch_objects"

    key = models.CharField(
        unique=True,
        blank=False,
        null=False,
        max_length=1024,
    )
