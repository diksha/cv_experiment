import typing as t

from loguru import logger

from core.portal.api.models.incident import Incident
from core.portal.lib.commands import CommandABC
from core.structs.incident import Incident as IncidentStruct


class UpdateIncident(CommandABC):
    def __init__(self, incident_struct: IncidentStruct):
        self.incident_struct = incident_struct

    def execute(self) -> Incident:
        """Update an incident from the provided incident struct.

        Returns:
            Incident:  updated incident

        Raises:
            Incident.DoesNotExist: if the incident does not exist
        """
        incident = Incident.objects_raw.get(uuid=self.incident_struct.uuid)
        existing_data_struct = IncidentStruct.from_dict(incident.data)
        new_data_struct = self.incident_struct

        incident.data[
            "end_frame_relative_ms"
        ] = self.merge_end_frame_relative_ms(
            existing_data_struct,
            new_data_struct,
        )

        incident.data[
            "original_end_frame_relative_ms"
        ] = self.get_original_end_frame_relative_ms(incident.data)

        incident.data["tail_incident_uuids"] = self.merge_tail_incident_uuids(
            existing_data_struct,
            new_data_struct,
        )

        incident.save(update_fields=["data"])
        return incident

    def merge_end_frame_relative_ms(
        self,
        existing_data_struct: IncidentStruct,
        new_data_struct: IncidentStruct,
    ) -> t.Optional[float]:
        """Merge new end timestamp with existing end timestamp.

        Args:
            existing_data_struct (IncidentStruct): existing incident data
            new_data_struct (IncidentStruct): new incident data

        Returns:
            t.Optional[float]: updated end timestamp if a valid timestamp
                exists, otherwise None
        """
        try:
            existing_value = float(existing_data_struct.end_frame_relative_ms)
        except (ValueError, TypeError):
            logger.error(
                "Invalid end_frame_relative_ms:"
                + f" {existing_data_struct.end_frame_relative_ms}"
            )
            existing_value = None

        try:
            new_value = float(new_data_struct.end_frame_relative_ms)
        except (ValueError, TypeError):
            logger.error(
                "Invalid end_frame_relative_ms:"
                + f" {new_data_struct.end_frame_relative_ms}"
            )
            new_value = None

        values = [
            value for value in [existing_value, new_value] if value is not None
        ]

        if values:
            return max(values)
        return None

    def merge_tail_incident_uuids(
        self,
        existing_data_struct: IncidentStruct,
        new_data_struct: IncidentStruct,
    ) -> t.List[str]:
        """Merge new tail incident UUIDs with existing UUIDs.

        Args:
            existing_data_struct (Incident): existing incident data
            new_data_struct (Incident): new incident data

        Returns:
            List[str]: list of tail incident UUIDs
        """
        existing_tail_incident_uuids = (
            existing_data_struct.tail_incident_uuids or []
        )
        new_tail_incident_uuids = new_data_struct.tail_incident_uuids or []
        return list(
            set(existing_tail_incident_uuids + new_tail_incident_uuids)
        )

    def get_original_end_frame_relative_ms(
        self, existing_data_dict: t.Dict[str, t.Any]
    ) -> t.Optional[float]:
        """Get the original end timestamp from the existing incident data.

        Args:
            existing_data_dict (IncidentStruct): existing incident data

        Returns:
            t.Optional[float]: original end timestamp if a valid timestamp
                exists, otherwise None
        """
        value = existing_data_dict.get(
            "original_end_frame_relative_ms",
            existing_data_dict.get("end_frame_relative_ms"),
        )

        try:
            return float(value)
        except (ValueError, TypeError):
            logger.error(f"Invalid original_end_frame_relative_ms: {value}")
        return None
