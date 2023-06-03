from django.db import models


class NotificationCategory(models.TextChoices):
    ORGANIZATION_DAILY_SUMMARY = (
        "ORGANIZATION_DAILY_SUMMARY",
        "Organization Daily Summary",
    )
    INCIDENT_ALERT = ("INCIDENT_ALERT", "Incident Alert")
    ZONE_PULSE = ("ZONE_PULSE", "Zone Pulse")
