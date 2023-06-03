from django.contrib import admin
from django.db import models

from core.portal.api.models.organization import Organization
from core.portal.devices.enums import EdgeLifecycle as EdgeLifecycleEnum
from core.portal.devices.models.edge_lifecycle import EdgeLifecycle
from core.portal.lib.models.base import Model


class Edge(Model):
    """An edge device that is used to process video streams.
    Properties:
        uuid (str): A unique identifier for the edge device.
        name (str): A user friendly name for the edge device.
        organization (Organization): The organization that the edge device belongs to.
        lifecycle (EdgeLifecycle): The status of the edge device.
        serial (str): The serial id of the edge device.
        mac_address (str): The mac address of the edge device.

    Methods:
        __str__(self): A string representation of the edge device.
    """

    class Meta:
        app_label = "devices"
        db_table = "edge"
        indexes = [
            models.Index(fields=["organization"]),
        ]

    lifecycle = models.ForeignKey(
        EdgeLifecycle,
        to_field="key",
        db_column="lifecycle",
        on_delete=models.RESTRICT,
        default=EdgeLifecycleEnum.EDGE_LIFECYCLE_LIVE.value,
        blank=False,
        null=False,
    )

    mac_address = models.CharField(
        help_text="MAC address of the edge device.",
        max_length=256,
        blank=True,
        null=True,
    )

    name = models.CharField(
        help_text="User friendly name displayed throughout apps.",
        max_length=256,
        null=False,
        blank=False,
    )

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        related_name="edges",
    )

    serial = models.CharField(
        help_text="Serial number of the edge device.",
        max_length=64,
        blank=True,
        null=True,
    )

    uuid = models.UUIDField(
        help_text="Unique identifier for the edge device.",
        blank=False,
        null=False,
        unique=True,
        auto_created=True,
    )

    def __str__(self) -> str:
        """Returns the string representation of the edge row.

        Returns:
            str: The string representation of the edge row.
        """
        return f"Edge uuid: {self.uuid}"


admin.site.register(Edge)
