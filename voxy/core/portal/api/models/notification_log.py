from django.contrib.auth.models import User
from django.db import models

from core.portal.api.models.incident import Incident
from core.portal.lib.models.base import Model
from core.portal.notifications.enums import NotificationCategory


class NotificationLog(Model):

    user = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    category = models.CharField(
        null=True,
        max_length=30,
        choices=NotificationCategory.choices,
    )
    incident = models.ForeignKey(
        Incident, null=True, on_delete=models.SET_NULL
    )
    data = models.JSONField(null=True)
    sent_at = models.DateTimeField()

    @property
    def from_utc(self):
        return (self.data or {}).get("from_utc", None)

    @property
    def to_utc(self):
        return (self.data or {}).get("to_utc", None)

    @property
    def group_by(self):
        return (self.data or {}).get("group_by", None)
