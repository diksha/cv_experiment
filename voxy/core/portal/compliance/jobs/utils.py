from typing import Dict

from django.db.backends.utils import CursorWrapper

from core.portal.api.models.organization import Organization
from core.portal.compliance.models.production_line import ProductionLine
from core.portal.devices.models.camera import Camera
from core.portal.zones.models.zone import Zone


def fetch_as_dict(cursor: CursorWrapper, params: Dict[str, str]):
    """Return all rows from a cursor as a dict.

    Args:
        cursor (CursorWrapper): database cursor
        params (Dict[str, str]): query params

    Returns:
        List[Dict[str, Any]]: query results
    """
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


class IDLookup:
    """Helper class for converting keys/UUIDs to database IDs."""

    def __init__(self) -> None:
        super().__init__()
        self.organizations = {}
        self.organization_key_id_map = {
            org.key.lower(): org.id for org in Organization.objects.all()
        }
        self.zone_key_id_map = {
            zone.key.lower(): zone.id for zone in Zone.objects.all()
        }
        self.camera_uuid_id_map = {
            camera.uuid.lower(): camera.id for camera in Camera.objects.all()
        }
        self.production_line_uuid_id_map = {
            production_line.uuid.lower(): production_line.id
            for production_line in ProductionLine.objects.all()
        }

    def get_organization_id(self, key: str) -> int:
        """Looks up organization ID from organization key.

        Args:
            key (str): organization key

        Returns:
            int: organization ID
        """
        organization_id = self.organization_key_id_map[key.lower()]
        return organization_id

    def get_zone_id(self, key: str) -> int:
        """Looks up zone ID from zone key.

        Args:
            key (str): zone key

        Returns:
            int: zone ID
        """
        zone_id = self.zone_key_id_map[key.lower()]
        return zone_id

    def get_camera_id(self, uuid: str) -> int:
        """Looks up camera ID from camera UUID.

        Args:
            uuid (str): camera UUID

        Returns:
            int: camera ID
        """
        camera_id = self.camera_uuid_id_map[uuid.lower()]
        return camera_id

    def get_production_line_id(self, uuid: str) -> int:
        """Looks up production line ID from production line UUID.

        Args:
            uuid (str): production line UUID

        Returns:
            int: production line ID
        """
        production_line_id = self.production_line_uuid_id_map[uuid.lower()]
        return production_line_id
