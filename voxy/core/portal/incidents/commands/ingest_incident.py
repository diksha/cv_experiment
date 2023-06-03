from django.db import IntegrityError
from loguru import logger

from core.portal.api.models.incident import Incident
from core.portal.incidents.commands.create_incident import CreateIncident
from core.portal.incidents.commands.update_incident import UpdateIncident
from core.portal.lib.commands import CommandABC
from core.structs.incident import Incident as IncidentStruct


class IngestIncident(CommandABC):
    def __init__(self, incident_struct: IncidentStruct):
        self.incident_struct = incident_struct

    def execute(self) -> Incident:
        """Ingest an incident from the provided incident struct.

        The incident ingestion process is intended to be "idempotent" in the
        sense that it can receive head/tail incident data out of order or
        receive the same incident data multiple times while maintaining the
        correct state in the portal database.

        Returns:
            Incident: created or updated incident
        """
        tail_incident_uuids = self.incident_struct.tail_incident_uuids or []
        is_updated_incident = len(tail_incident_uuids) > 0

        if is_updated_incident:
            return self._ingest_updated_incident()
        return self._ingest_new_incident()

    def _ingest_new_incident(self) -> Incident:
        """Ingest a new incident from the provided data dictionary.

        Returns:
            Incident: created incident
        """
        try:
            return CreateIncident(self.incident_struct).execute()
        except IntegrityError:
            logger.warning(
                "Received create request for incident which"
                + f" already exists: {self.incident_struct.to_dict()}"
            )
            return UpdateIncident(self.incident_struct).execute()

    def _ingest_updated_incident(self) -> Incident:
        """Ingest an updated incident from the provided data dictionary.

        Returns:
            Incident:  updated incident
        """
        try:
            return UpdateIncident(self.incident_struct).execute()
        except Incident.DoesNotExist:
            logger.warning(
                "Received update request for non-existent"
                + f" incident: {self.incident_struct.to_dict()}"
            )
            return CreateIncident(self.incident_struct).execute()
