import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone

from django.contrib.auth.models import User
from django.db import transaction

from core.portal.api.models.incident import Incident
from core.portal.api.models.incident_feedback import IncidentFeedback
from core.portal.incident_feedback.enums import IncidentFeedbackType
from core.portal.incident_feedback.lock import unlock_incident
from core.portal.incidents.visibility import sync_incident_visibility


@dataclass
class CreateIncidentFeedbackService:
    user: User
    incident: Incident
    value: str
    comment: t.Optional[str] = None
    elapsed_milliseconds_between_reviews: t.Optional[int] = None
    incident_served_timestamp_seconds: t.Optional[int] = None

    @property
    def feedback_count_fields_to_increment(self) -> t.List[str]:
        """Get list of feedback count fields which need to be incremented.

        Returns:
            t.List[str]: list of field names
        """
        if self.value == "valid":
            return ["valid_feedback_count"]
        if self.value == "invalid":
            return ["invalid_feedback_count"]
        if self.value == "unsure":
            return ["unsure_feedback_count"]
        if self.value == "corrupt":
            return ["corrupt_feedback_count", "invalid_feedback_count"]
        return []

    def execute(self) -> IncidentFeedback:
        """Create incident feedback and handle related side effects.

        Returns:
            IncidentFeedback: created feedback
        """
        with transaction.atomic():
            feedback = IncidentFeedback.objects.create(
                user=self.user,
                incident=self.incident,
                feedback_type=IncidentFeedbackType.ACCURACY.value,
                feedback_value=self.value,
                feedback_text=self.comment,
                elapsed_milliseconds_between_reviews=self.elapsed_milliseconds_between_reviews,
                incident_served_timestamp_seconds=self.incident_served_timestamp_seconds,
            )
            feedback.incident.last_feedback_submission_timestamp = (
                feedback.created_at
            )

            for field in self.feedback_count_fields_to_increment:
                setattr(
                    feedback.incident,
                    field,
                    getattr(feedback.incident, field) + 1,
                )

            feedback.incident.save(
                update_fields=[
                    "last_feedback_submission_timestamp",
                    *self.feedback_count_fields_to_increment,
                ]
            )

            sync_incident_visibility(feedback.incident)
            unlock_incident(feedback.incident.pk)

        return feedback


@dataclass
class CreateShadowIncidentFeedbackService:
    user: User
    incident: Incident
    value: str
    comment: t.Optional[str] = None
    elapsed_milliseconds_between_reviews: t.Optional[int] = None
    incident_served_timestamp_seconds: t.Optional[int] = None

    def execute(self):
        """Create incident feedback for shadow review and store in data JSON field."""
        with transaction.atomic():
            data = self.incident.data or {}
            data["shadow_reviewed"] = True

            review = {
                "feedback_value": self.value,
                "user_id": self.user.id,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }

            if self.comment:
                review["comments"] = self.comment

            if self.elapsed_milliseconds_between_reviews:
                review[
                    "elapsed_milliseconds_between_reviews"
                ] = self.elapsed_milliseconds_between_reviews

            if self.incident_served_timestamp_seconds:
                review[
                    "incident_served_timestamp_seconds"
                ] = self.incident_served_timestamp_seconds

            if "shadow_reviews" not in data:
                data["shadow_reviews"] = []

            data["shadow_reviews"].append(review)

            self.incident.data = data
            self.incident.save(update_fields=["data"])
            unlock_incident(self.incident.id)
