# trunk-ignore-all(bandit/B311): don't need cryptographtic randomness
import random
import typing as t
from dataclasses import dataclass

from loguru import logger

from core.portal.api.models.incident import Incident
from core.portal.api.models.notification_log import NotificationLog
from core.portal.incidents.alerts import AlertManager
from core.portal.incidents.enums import IncidentTypeKey, ReviewStatus
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.notifications.enums import NotificationCategory

COOLDOWN_VALID_REVIEW_COUNT = 1


@dataclass
class ReviewLevelConfig:
    minimum_valid_reviews: int
    gating: bool
    sample_rate: t.Optional[float] = None


REVIEW_LEVEL_CONFIGS = {
    # Gating, 100% of incidents should be reviewed
    ReviewLevel.RED: ReviewLevelConfig(2, True),
    # Gating, 100% of incidents should be reviewed
    ReviewLevel.YELLOW: ReviewLevelConfig(1, True),
    # Non-gating, 100% of incidents should be reviewed
    ReviewLevel.GREEN: ReviewLevelConfig(1, False, sample_rate=1.0),
    # Non-gating, 1% of incidents should be reviewed
    ReviewLevel.GOLD: ReviewLevelConfig(1, False, sample_rate=0.01),
}


def have_notifications_been_sent(incident: Incident) -> bool:
    """Determine if notifications have been sent for the provided incident.

    Args:
        incident (Incident): incident

    Returns:
        bool: true if notifications have been sent, othewise false
    """
    return NotificationLog.objects.filter(
        incident=incident,
        category=NotificationCategory.INCIDENT_ALERT,
    ).exists()


def maybe_show_incident_to_customers(incident: Incident) -> None:
    """Handles showing incidents to customers and related side effects.

    Args:
        incident (Incident): incident
    """

    # Do not show or alert on cooldown incidents
    if incident.is_cooldown:
        return

    # If incident is currently hidden from customers, make it visible
    if not incident.visible_to_customers:
        incident.visible_to_customers = True
        incident.save()
        AlertManager(incident).maybe_send_alert()


def maybe_hide_incident_from_customers(
    incident: Incident,
    notifications_sent: t.Optional[bool] = None,
) -> None:
    """Handle hiding incidents from customers and related side effects.

    Args:
        incident (Incident): incident
        notifications_sent(t.Optional[bool]): true if notifications have
            been sent for this incident, otherwise false
    """
    notifications_sent = (
        notifications_sent
        if notifications_sent is not None
        else have_notifications_been_sent(incident)
    )
    if notifications_sent:
        logger.warning(
            "Not hiding incident from customers because alert notifications"
            + f" have already been sent (UUID: {incident.uuid})"
        )
        return

    # If incident is currently visible to customers, hide it
    if incident.visible_to_customers:
        incident.visible_to_customers = False
        incident.save()


def set_review_status(incident: Incident, review_status: ReviewStatus) -> None:
    """Set incident review status (if not already set).

    Args:
        incident (Incident): incident
        review_status (ReviewStatus): desired review status
    """
    if incident.review_status != review_status:
        incident.review_status = review_status
        incident.save()


def is_ignored_incident(incident: Incident) -> bool:
    """Checks if incident should be ignored and marked as does not need review

    Args:
        incident (Incident): incident

    Returns:
        bool: true if cooldown incident of certain incident type
    """
    return incident.is_cooldown and incident.incident_type.key in [
        IncidentTypeKey.SPILL,
        IncidentTypeKey.PRODUCTION_LINE_DOWN,
        IncidentTypeKey.OPEN_DOOR_DURATION,
    ]


def sync_incident_visibility(incident: Incident) -> None:
    """Sync incident visibility properties with current incident state.

    There are 3 primary triggers for this function:
        - incident created
        - incident feedback created
        - incident feedback deleted

    However, this function is intended to be idempotent so it can be
    called at any time and ensure the incident visibility properties
    are correct based on the current state of this incident.

    Args:
        incident (Incident): incident instance
    """

    config = REVIEW_LEVEL_CONFIGS.get(incident.review_level)

    if not config:
        logger.error(
            f"Invalid review level ({incident.review_level}) for"
            + f" incident UUID: {incident.uuid}"
        )
        return

    minimum_valid_reviews = (
        min(config.minimum_valid_reviews, COOLDOWN_VALID_REVIEW_COUNT)
        if incident.is_cooldown
        else config.minimum_valid_reviews
    )

    notifications_sent = have_notifications_been_sent(incident)

    if notifications_sent:
        # If notifications have been sent, mark as valid to prevent
        # users from clicking a notification link and getting a 404
        set_review_status(incident, ReviewStatus.VALID)
    elif incident.invalid_feedback_count > 0:
        # Any invalid feedback should invalidate the
        # incident and hide it from customers
        set_review_status(incident, ReviewStatus.INVALID)
        maybe_hide_incident_from_customers(incident, notifications_sent)
    elif incident.valid_feedback_count >= minimum_valid_reviews:
        # Once the minimum number of valid reviews has been reached,
        # mark the incident as valid and show it to customers
        set_review_status(incident, ReviewStatus.VALID)
        maybe_show_incident_to_customers(incident)
    elif incident.unsure_feedback_count > 0 or is_ignored_incident(incident):
        # Any unsure feedback received before the incident is considered valid
        # or invalid should both hide the incident and prevent further reviews
        set_review_status(incident, ReviewStatus.DO_NOT_REVIEW)
        maybe_hide_incident_from_customers(incident, notifications_sent)
    else:
        # If no feedback has been received, determine the initial review status
        if config.gating:
            # For gating review levels, mark as needing review and
            # hide from customers
            set_review_status(incident, ReviewStatus.NEEDS_REVIEW)
            maybe_hide_incident_from_customers(incident, notifications_sent)
        elif config.sample_rate is None:
            # For non-gating review levels with no sample rate defined,
            # mark as valid and show to customers
            set_review_status(incident, ReviewStatus.VALID)
            maybe_show_incident_to_customers(incident)
        else:
            # For non-gating review levels, we want to use the config sample rate
            # to randomly select a subset of incidents to get reviewed
            review_status = (
                ReviewStatus.VALID_AND_NEEDS_REVIEW
                if random.random() <= config.sample_rate
                else ReviewStatus.VALID
            )
            set_review_status(incident, review_status)
            maybe_show_incident_to_customers(incident)
