import pytest

from core.portal.incident_feedback.enums import IncidentAccuracyOption
from core.portal.incident_feedback.services import (
    CreateIncidentFeedbackService,
    CreateShadowIncidentFeedbackService,
)
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.testing.factories import IncidentFactory, UserFactory


@pytest.mark.django_db
def test_invalid_feedback_hides_incident() -> None:
    """Test that receiving invalid feedback hides incident."""
    user = UserFactory()
    incident = IncidentFactory(visible_to_customers=True)
    CreateIncidentFeedbackService(
        user,
        incident,
        IncidentAccuracyOption.INVALID.value,
    ).execute()
    incident.refresh_from_db()
    assert incident.visible_to_customers is False


@pytest.mark.django_db
def test_yellow_incident_visible_after_one_valid_feedback() -> None:
    """Test that yellow incidents are visible after one valid feedback."""
    user = UserFactory()
    incident = IncidentFactory(review_level=ReviewLevel.YELLOW)
    assert incident.visible_to_customers is False

    CreateIncidentFeedbackService(
        user,
        incident,
        IncidentAccuracyOption.VALID.value,
    ).execute()
    incident.refresh_from_db()
    assert incident.visible_to_customers is True


@pytest.mark.django_db
def test_red_incident_visible_after_two_valid_feedback() -> None:
    """Test that red incidents are visible after two valid feedbacks."""
    user = UserFactory()
    incident = IncidentFactory(review_level=ReviewLevel.RED)
    assert incident.visible_to_customers is False

    CreateIncidentFeedbackService(
        user,
        incident,
        IncidentAccuracyOption.VALID.value,
    ).execute()
    assert incident.visible_to_customers is False

    CreateIncidentFeedbackService(
        user,
        incident,
        IncidentAccuracyOption.VALID.value,
    ).execute()
    incident.refresh_from_db()
    assert incident.visible_to_customers is True


@pytest.mark.django_db
def test_mark_as_shadow_reviewed_sets_data_fields() -> None:
    """Test that mark_as_shadow_reviewed sets the appropriate data fields."""
    user = UserFactory()
    incident = IncidentFactory()

    feedback_service = CreateShadowIncidentFeedbackService(
        user,
        incident,
        IncidentAccuracyOption.VALID.value,
        comment="Sample comment",
        elapsed_milliseconds_between_reviews=1234,
    )
    feedback_service.execute()
    incident.refresh_from_db()

    assert incident.data["shadow_reviewed"] is True
    assert (
        incident.data["shadow_reviews"][0]["feedback_value"]
        == IncidentAccuracyOption.VALID.value
    )
    assert incident.data["shadow_reviews"][0]["comments"] == "Sample comment"
    assert (
        incident.data["shadow_reviews"][0][
            "elapsed_milliseconds_between_reviews"
        ]
        == 1234
    )

    # Test that does not update incident visibility or feedback counts
    assert incident.visible_to_customers is False
    assert incident.valid_feedback_count == 0
    assert incident.invalid_feedback_count == 0
    assert incident.unsure_feedback_count == 0
    assert incident.corrupt_feedback_count == 0
