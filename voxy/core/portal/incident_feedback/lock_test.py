import pytest
from django.core.cache import cache

from core.portal.incident_feedback.lock import (
    get_locked_incident_ids,
    lock_incident,
    unlock_incident,
)


@pytest.mark.django_db
def test_review_lock_happy_path() -> None:
    """Test core lock operation happy paths work as expected."""
    # Clear existing locks
    cache.clear()

    # Lock a few incidents
    lock_incident(1)
    lock_incident(2)
    lock_incident(3)
    lock_incident(4)
    lock_incident(5)

    # Verify lock contains expected values
    locked_incident_ids = get_locked_incident_ids()
    assert locked_incident_ids == {1, 2, 3, 4, 5}

    # Unlock a few incidents
    unlock_incident(2)
    unlock_incident(4)

    # Verify lock contains expected values
    locked_incident_ids = get_locked_incident_ids()
    assert locked_incident_ids == {1, 3, 5}
