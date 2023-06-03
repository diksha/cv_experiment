import typing as t
from dataclasses import dataclass
from functools import cached_property
from uuid import UUID


@dataclass
class DemoSourceIncident:
    url: str
    highlighted: bool = False


@dataclass
class DemoIncidentTypeConfig:
    incident_type_key: str
    source_incidents: t.List[DemoSourceIncident]
    relative_day_config: t.Dict[int, int]

    def __post_init__(self):
        self.source_uuid_queue = InfiniteQueue(
            list(self.source_incident_uuids)
        )

    def _parse_uuid_from_url(self, url: str) -> str:
        return url.split("/")[-1]

    def is_valid(self) -> bool:
        """Check if the config is valid.

        Returns:
            bool: true if config is valid, otherwise false
        """
        for uuid in self.source_incident_uuids:
            try:
                UUID(uuid)
            except ValueError:
                return False
        return True

    @cached_property
    def source_incident_uuids(self) -> t.Set[str]:
        """Set of unique source incident uuids.

        Returns:
            t.Set[str]: set of unique source incident uuids
        """
        return {
            self._parse_uuid_from_url(si.url) for si in self.source_incidents
        }


@dataclass
class InfiniteQueue:
    values: t.List[t.Any]

    def get_value(self) -> t.Any:
        """Get the next value in the queue.

        Returns:
            t.Any: next value in the queue
        """
        value = self.values.pop(0)
        self.values.append(value)
        return value
