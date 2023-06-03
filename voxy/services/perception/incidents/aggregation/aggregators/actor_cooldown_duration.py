import typing as t

from cachetools import TTLCache
from loguru import logger

from core.structs.incident import Incident


class ActorCooldownDurationAggregator:
    @classmethod
    def _build_aggregation_key(cls, incident: Incident) -> t.Optional[str]:
        """Build an aggregation key for the provided incident.

        Aggregation key should be a unique identifier that is consistent
        across all "chunks" of the same logical incident.

        Args:
            incident (Incident): incident

        Returns:
            str: aggregation key if mergeable, None otherwise
        """
        key_parts = [
            incident.run_uuid,
            incident.camera_uuid,
            incident.track_uuid,
        ]

        is_valid_key = all(item is not None for item in key_parts)
        return ":".join(map(str, key_parts)).lower() if is_valid_key else None

    def __init__(
        self,
        max_tail_incident_lag_seconds: int = 3600,
        max_cached_incident_count: int = 10000,
        epsilon_between_incidents_ms: int = 10000,
    ):
        """Constructor.

        Args:
            max_tail_incident_lag_seconds (int, optional):
                Maximum seconds to cache and wait for tail incident. Defaults to 3600.
            max_cached_incident_count (int, optional):
                Maximum incidents to cache. Defaults to 10000.
            epsilon_between_incidents_ms (int, optional):
                Acceptable difference in milliseconds between two incidents for aggregation.
                Defaults to 10000.
        """

        self._head_incident_cache = TTLCache(
            max_cached_incident_count,
            max_tail_incident_lag_seconds,
        )
        self._epsilon_between_incidents_ms = epsilon_between_incidents_ms

    def aggregate(self, incident: Incident) -> Incident:
        """Aggregate the provided incident into the head incident.

        Args:
            incident (Incident): incident

        Returns:
            Incident: processed incidents
        """
        aggregation_key = self._build_aggregation_key(incident)

        if not aggregation_key:
            logger.warning(
                f"Invalid aggregation key for incident: {incident.uuid}"
            )
            return incident

        head_incident = self._head_incident_cache.get(aggregation_key)

        if (
            not head_incident
            or not incident.cooldown_tag
            or head_incident.end_frame_relative_ms
            + self._epsilon_between_incidents_ms
            < incident.start_frame_relative_ms
        ):
            # This is a new incident, update cache and return it
            self._head_incident_cache[aggregation_key] = incident
            return incident

        # Update end timestamp
        head_incident.end_frame_relative_ms = max(
            head_incident.end_frame_relative_ms,
            incident.end_frame_relative_ms,
        )

        # Keep track of tail incident UUIDs, just in case we need them later
        tail_incident_uuids = head_incident.tail_incident_uuids or []
        tail_incident_uuids.append(incident.uuid)
        head_incident.tail_incident_uuids = tail_incident_uuids

        # Refresh head incident cache entry (which also resets TTL)
        self._head_incident_cache[aggregation_key] = head_incident
        return head_incident
