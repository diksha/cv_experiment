# trunk-ignore-all(pylint/C0413,flake8/E402): allow django setup() before model imports

import django
from loguru import logger

django.setup()

from core.portal.api.models.incident_type import IncidentType
from core.portal.perceived_data.models.perceived_event_rate_definition import (
    PerceivedEventRateDefinition,
)
from services.portal.configs.perceived_event_rate_definitions import (
    definitions,
)


class SyncPerceivedEventRateDefinitions:
    def run(self):
        """Run the sync process."""

        incident_type_map = {it.key: it for it in IncidentType.objects.all()}

        objects_to_sync = []
        for (
            incident_type_key,
            incident_type_definitions,
        ) in definitions.items():
            for definition in incident_type_definitions:
                incident_type = incident_type_map.get(incident_type_key)
                if not incident_type:
                    logger.error(
                        f"Unknown incident type key: {incident_type_key}"
                    )
                definition.incident_type = incident_type
                objects_to_sync.append(definition)

        results = PerceivedEventRateDefinition.objects.bulk_create(
            objects_to_sync,
            update_conflicts=True,
            unique_fields=[
                "incident_type_id",
                "calculation_method",
                "perceived_actor_state_duration_category",
            ],
            update_fields=["name"],
        )
        logger.info(f"Synced {len(results)} records")


if __name__ == "__main__":
    SyncPerceivedEventRateDefinitions().run()
