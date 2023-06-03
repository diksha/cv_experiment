import typing as t

from django.core.cache import cache

REVIEW_LOCK_DELIMITER = ":"
REVIEW_LOCK_PREFIX = (
    f"REVIEW{REVIEW_LOCK_DELIMITER}LOCK{REVIEW_LOCK_DELIMITER}"
)
REVIEW_LOCK_WILDCARD = f"{REVIEW_LOCK_PREFIX}*"
REVIEW_LOCK_EXPIRATION_SECONDS = 600


def _get_review_lock_key(incident_id: int) -> str:
    """Generate the review lock key for the given incident_id.

    Args:
        incident_id (int): incident ID

    Returns:
        str: lock key
    """
    return f"{REVIEW_LOCK_PREFIX}{incident_id}"


def lock_incident(
    incident_id: int,
    expires_in: t.Optional[int] = REVIEW_LOCK_EXPIRATION_SECONDS,
) -> bool:
    """Add an incident to the review lock.

    Args:
        incident_id (int): incident ID
        expires_in (t.Optional[int]): time to expire of the
            provided incident ID

    Returns:
        bool: whether it was able to acquire a lock or not
    """
    lock_key = _get_review_lock_key(incident_id)

    return cache.set(lock_key, incident_id, nx=True, timeout=expires_in)


def unlock_incident(incident_id: int) -> None:
    """Remove an incident ID from the review lock.

    Args:
        incident_id (int): incident ID
    """
    lock_key = _get_review_lock_key(incident_id)
    cache.delete(lock_key)


def get_locked_incident_ids() -> t.Set[int]:
    """Get locked incident IDs.

    Returns:
        t.Set[int]: set of incident IDs
    """
    incident_ids = set()

    for key in cache.iter_keys(REVIEW_LOCK_WILDCARD):
        # Extract the incident_id from the key and add it to the set
        parts = key.split(REVIEW_LOCK_DELIMITER)
        incident_id = int(parts[-1])
        incident_ids.add(incident_id)

    return incident_ids
