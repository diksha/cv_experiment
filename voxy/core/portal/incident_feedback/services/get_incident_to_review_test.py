from typing import List

import pytest
from django.core.cache import cache

from core.portal.incident_feedback.services import (
    get_incident_to_review_for_shadow_reviewers,
)
from core.portal.testing.factories import (
    IncidentFactory,
    IncidentTypeFactory,
    UserFactory,
)


@pytest.mark.django_db
def test_get_incident_for_shadow_reviewers_eligible_incident() -> None:
    """Test that the function returns an eligible incident for shadow review."""
    cache.clear()

    user = UserFactory()
    excluded_incident_uuids: List[str] = []

    # Create eligible incidents for shadow review
    incident = IncidentFactory(
        cooldown_source=None,
        valid_feedback_count=1,
        incident_type=IncidentTypeFactory(key="foo"),
        data={},
    )

    result = get_incident_to_review_for_shadow_reviewers(
        user,
        excluded_incident_uuids,
    )
    assert result == incident


@pytest.mark.django_db
def test_get_incident_for_shadow_reviewers_exclude_shadow_reviewed_incidents() -> None:
    """Test that the function excludes incidents that have already been shadow reviewed."""
    user = UserFactory()
    excluded_incident_uuids: List[str] = []

    # Create incidents with shadow_reviewed flag set to True
    IncidentFactory(
        cooldown_source=None,
        valid_feedback_count=1,
        data={"shadow_reviewed": True},
    )

    IncidentFactory(
        cooldown_source=None,
        valid_feedback_count=1,
        data={"shadow_reviewed": True},
    )

    result = get_incident_to_review_for_shadow_reviewers(
        user,
        excluded_incident_uuids,
    )
    assert result is None
