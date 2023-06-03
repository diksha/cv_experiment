from django.contrib import admin
from django.contrib.auth.models import User
from django.db import models

from core.portal.api.models.incident import Incident
from core.portal.lib.models.base import Model


class IncidentFeedback(Model):
    """Represents user feedback for a particular incident.

    There is little validation/structure here to allow us to easily implement
    new forms of feedback. Once the shape of feedback is better understood we
    should add more validation/structure, get types and values from enums, etc.

    Expected feedback types:

    1. Incident accuracy (do the incident details align with what's seen in the video?)
        - Type: incident_accuracy
        - Values: valid, invalid, unsure, corrupt
    """

    incident = models.ForeignKey(
        Incident, on_delete=models.CASCADE, related_name="feedback"
    )
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="incident_feedback"
    )
    feedback_type = models.CharField(max_length=100)
    feedback_value = models.CharField(max_length=100)
    feedback_text = models.TextField(null=True, blank=True)

    incident_served_timestamp_seconds = models.IntegerField(
        null=True, blank=True
    )
    elapsed_milliseconds_between_reviews = models.IntegerField(
        null=True, blank=True
    )


class IncidentFeedbackAdmin(admin.ModelAdmin):
    list_display = (
        "incident",
        "user",
        "feedback_type",
        "feedback_value",
        "feedback_text",
    )
