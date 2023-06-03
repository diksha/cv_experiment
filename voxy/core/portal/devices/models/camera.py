from typing import Optional

from django.contrib import admin
from django.db import models

from core.portal.api.models.organization import Organization
from core.portal.devices.models.edge import Edge
from core.portal.lib.models.base import Model
from core.portal.lib.utils.signed_url_manager import signed_url_manager
from core.portal.zones.models.zone import Zone


# TODO(PORTAL-166): get mypy working with model fields
class Camera(Model):
    class Meta:
        app_label = "devices"
        db_table = "camera"
        indexes = [
            models.Index(fields=["organization"]),
            models.Index(fields=["zone"]),
        ]

    uuid = models.CharField(
        max_length=250, null=False, blank=False, unique=True
    )
    name = models.CharField(
        help_text="User friendly name displayed throughout apps.",
        max_length=250,
        null=False,
        blank=False,
    )
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        related_name="cameras",
    )
    zone = models.ForeignKey(
        Zone,
        on_delete=models.RESTRICT,
        blank=True,
        null=True,
        related_name="cameras",
    )

    edge = models.ForeignKey(
        Edge,
        on_delete=models.RESTRICT,
        blank=True,
        null=True,
    )

    # TODO(PRO-596): delete this field
    thumbnail_gcs_path = models.CharField(
        max_length=250, null=True, blank=True
    )

    thumbnail_s3_path = models.CharField(max_length=250, null=True, blank=True)

    @property
    def thumbnail_url(self) -> Optional[str]:
        """Property which defines the thumbnail url used by camera.

        Returns:
            Optional[str]: signed thumbnail url for camera
        """
        if self.thumbnail_s3_path:
            return signed_url_manager.get_signed_url(
                s3_path=self.thumbnail_s3_path,
            )

        most_recent_incident = self.incidents.order_by("-timestamp").last()

        if most_recent_incident:
            if most_recent_incident.video_thumbnail_s3_path:
                self.thumbnail_s3_path = (
                    most_recent_incident.video_thumbnail_s3_path
                )
                self.save()

                return signed_url_manager.get_signed_url(
                    s3_path=most_recent_incident.video_thumbnail_s3_path,
                )

        return None

    def __str__(self) -> str:
        """__str__ function

        Returns:
            str: string representation.
        """
        return f"Camera uuid: {self.uuid}"

    @property
    def enabled_incident_types(self):
        """Returns enabled incident types for the zone
        Returns:
            CameraIncidentType: enabled camera incident types
        """
        return self.camera_incident_types.select_related(
            "incident_type"
        ).filter(
            enabled=True,
        )


class CameraConfigNew(Model):
    class Meta:
        app_label = "devices"
        db_table = "cameraconfignew"
        unique_together = (
            "camera",
            "version",
        )

    camera = models.ForeignKey(
        Camera, blank=False, null=False, on_delete=models.CASCADE
    )
    doors = models.JSONField(blank=True, null=True)
    driving_areas = models.JSONField(blank=True, null=True)
    actionable_regions = models.JSONField(blank=True, null=True)
    intersections = models.JSONField(blank=True, null=True)
    end_of_aisles = models.JSONField(blank=True, null=True)
    no_pedestrian_zones = models.JSONField(blank=True, null=True)
    motion_detection_zones = models.JSONField(blank=True, null=True)
    no_obstruction_regions = models.JSONField(blank=True, null=True)
    version = models.IntegerField(default=0)

    def __str__(self) -> str:
        """__str__ function

        Returns:
            str: string representation.
        """
        return f"CameraConfigNew uuid: {self.camera}  version {self.version}"


admin.site.register(CameraConfigNew)
admin.site.register(Camera)
