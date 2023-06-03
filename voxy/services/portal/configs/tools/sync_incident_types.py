# trunk-ignore-all(pylint/C0413,flake8/E402): allow django setup() before model imports

import django
from loguru import logger

django.setup()

from core.portal.api.models.incident_type import IncidentType
from services.portal.configs.incident_types import incident_types


class SyncIncidentTypes:
    def run(self):
        """Run the sync process."""
        results = IncidentType.objects.bulk_create(
            incident_types.values(),
            update_conflicts=True,
            unique_fields=["key"],
            update_fields=["name", "category", "background_color"],
        )
        logger.info(f"Synced {len(results)} records")


if __name__ == "__main__":
    SyncIncidentTypes().run()
