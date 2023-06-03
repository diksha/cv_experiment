from datetime import datetime, timezone

from core.portal.api.models.incident import Incident
from core.portal.api.models.incident_type import (
    IncidentType,
    OrganizationIncidentType,
)
from core.portal.api.models.organization import Organization
from core.portal.devices.models.camera import Camera
from core.portal.incidents.enums import CooldownSource, IncidentTypeKey
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.incidents.visibility import sync_incident_visibility
from core.portal.lib.commands import CommandABC
from core.structs.incident import Incident as IncidentStruct


class CreateIncident(CommandABC):
    def __init__(self, incident_struct: IncidentStruct):
        self.incident_struct = incident_struct

    def execute(self) -> Incident:
        """Create an incident from the provided incident struct.

        Returns:
            Incident: created incident
        """
        organization = Organization.objects.get(
            key=self.incident_struct.organization_key
        )

        incident_type = IncidentType.objects.filter(
            key=self.incident_struct.incident_type_id
        ).first()

        organization_incident_type = OrganizationIncidentType.objects.filter(
            organization=organization, incident_type=incident_type
        ).first()

        review_level = (
            organization_incident_type.review_level
            if organization_incident_type
            else ReviewLevel.RED
        )

        camera = Camera.objects.filter(
            uuid=self.incident_struct.camera_uuid
        ).first()

        site = camera.zone if camera and camera.zone else None
        site_config = site.config if site and site.config else {}

        # HACK: force production line down incidents to be non-cooldown
        #       at certain sites
        force_non_cooldown = (
            incident_type
            and incident_type.key == IncidentTypeKey.PRODUCTION_LINE_DOWN
            and site_config.get("force_production_line_down_as_non_cooldown")
        )

        cooldown_tag = (
            (self.incident_struct.cooldown_tag or "False").strip().lower()
        )
        is_cooldown_incident = cooldown_tag != "false"
        cooldown_source = (
            CooldownSource.COOLDOWN
            if is_cooldown_incident and not force_non_cooldown
            else None
        )

        # Capture timestamp (default to now if relative_ms not present)
        try:
            start_frame_relative_ms = float(
                self.incident_struct.start_frame_relative_ms
            )
            start_frame_relative_s = start_frame_relative_ms / 1000.0
            timestamp = datetime.fromtimestamp(
                start_frame_relative_s, tz=timezone.utc
            )
        except (ValueError, TypeError):
            timestamp = datetime.now(tz=timezone.utc)

        incident = Incident.objects.create(
            uuid=self.incident_struct.uuid,
            title=self.incident_struct.title,
            incident_type=incident_type,
            organization=organization,
            timestamp=timestamp,
            priority=self.incident_struct.priority or Incident.Priority.MEDIUM,
            status=Incident.Status.OPEN,
            highlighted=False,
            experimental=Incident.is_experimental_version(
                self.incident_struct.incident_version
            ),
            review_level=review_level,
            cooldown_source=cooldown_source,
            camera=camera,
            zone=camera.zone if camera else None,
            # Capture all data from the request body
            data=self.incident_struct.to_dict(),
        )

        sync_incident_visibility(incident)

        return incident
