from django.db import models


class ComplianceTypeKey(models.TextChoices):
    """All available compliance type keys."""

    SAFETY_VEST_STATE = ("SAFETY_VEST_STATE", "Safety Vest")
    HARD_HAT_STATE = ("HARD_HAT_STATE", "Hard Hat")
    POSTURE_EVENTS = ("POSTURE_EVENTS", "Posture")
    DOOR_OPEN_CLOSED_STATE = ("DOOR_OPEN_CLOSED_STATE", "Door Open/Closed")
    DOOR_CLOSED_WITHIN_30S_EVENTS = (
        "DOOR_CLOSED_WITHIN_30S_EVENTS",
        "Door Closed Within 30 Seconds",
    )
    DOOR_OPEN_DURATION = ("DOOR_OPEN_DURATION", "Door Open Duration")
