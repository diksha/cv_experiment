import datetime
import uuid

from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

from core.portal.api.models.incident import Incident
from core.portal.lib.models.base import Model

SHARELINK_EXPIRED = 3  # in days


class ShareLink(Model):
    token = models.CharField(max_length=100, null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    shared_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="user_shared_by",
    )

    visits = models.IntegerField(
        null=False,
        default=0,
    )

    incident = models.ForeignKey(
        Incident,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
    )

    def increment_visits(self):
        """Increment visits count for share link."""
        self.visits += 1
        self.save()

    def invalidate(self) -> None:
        """Invalidates the share link."""
        self.expires_at = timezone.now()
        self.save()

    @staticmethod
    def generate(incident: Incident, shared_by: User) -> str:
        """Helper method to generate a new share link.

        Args:
            incident (Incident): the incident we are creating shareable links for
            shared_by (User): the user sharing the link

        Returns:
            str: formatted url of share link
        """
        token = uuid.uuid4()
        share_link = ShareLink.objects.create(
            token=token,
            shared_by=shared_by,
            incident=incident,
            expires_at=timezone.now()
            + datetime.timedelta(days=SHARELINK_EXPIRED),
        )

        return f"{settings.BASE_URL}/share/{share_link.token}"
