import datetime
import uuid
from typing import List

from django.conf import settings
from django.contrib.auth.models import User
from django.db import IntegrityError, models, transaction
from django.utils import timezone

from core.portal.accounts.models.role import Role
from core.portal.api.models.organization import Organization
from core.portal.lib.models.base import Model
from core.portal.notifications.clients.sendgrid import (
    USER_INVITATION_TEMPLATE_ID,
    SendGridClient,
)
from core.portal.zones.models.zone import Zone

INVITATION_EXPIRED = 7  # in days


class Invitation(Model):
    invitee = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    organization = models.ForeignKey(
        Organization, null=True, on_delete=models.CASCADE
    )
    token = models.CharField(max_length=100, null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    invited_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="user_invited_by",
    )

    role = models.ForeignKey(
        Role,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )

    redeemed = models.BooleanField(
        null=True,
        blank=True,
    )

    zones = models.ManyToManyField(
        Zone,
        blank=True,
    )

    def invalidate(self) -> None:
        """Invalidates the invitation."""
        self.expires_at = timezone.now()
        self.save()

    def resend(self, invited_by: User) -> None:
        """Resends the invitation.

        Args:
            invited_by (User): user sending the invite
        """
        Invitation.send(
            invited_by=invited_by,
            email=self.invitee.email,
            role=self.role,
            organization=self.organization,
            zones=self.zones.all(),
        )

    @staticmethod
    def send(
        invited_by: User,
        email: str,
        role: Role,
        organization: Organization,
        zones: List[Zone],
    ) -> None:
        """Sends invitation.

        Args:
            invited_by (User): user sending the invite
            email (str): the email of the user that is being invited
            role (Role): the role associated with the invitation
            organization (Organization): the organization that the user is invited to
            zones (List[Zone]): a list of zones associated with the invitation


        Raises:
            RuntimeError: if there is an existing user with a redeemed invitation
        """

        with transaction.atomic():
            existing_invitations = Invitation.objects.filter(
                invitee__email=email
            )
            for invitation in existing_invitations:
                if invitation.redeemed:
                    raise RuntimeError(
                        "Cannot reinvite a user who has already redeemed an invitation."
                    )
                if invitation.expires_at >= timezone.now():
                    raise RuntimeError(
                        "Cannot reinvite a user who has a valid invitation."
                    )
                # Invalidate all existing expired and unredeemed invitations
                invitation.invalidate()

            if existing_invitations.exists():
                new_user = User.objects.get(email=email)
            else:
                try:
                    new_user = User(
                        email=email,
                        username=email,
                        is_staff=False,
                        is_superuser=False,
                        is_active=False,
                    )
                    new_user.save()
                except IntegrityError as duplicate_error:
                    raise RuntimeError(
                        "The user with this email has already been registered"
                    ) from duplicate_error

            token = uuid.uuid4()
            invitation = Invitation(
                token=token,
                expires_at=timezone.now()
                + datetime.timedelta(days=INVITATION_EXPIRED),
                invited_by=invited_by,
                invitee=new_user,
                role=role,
                organization=organization,
                redeemed=False,
            )
            invitation.save()
            invitation.zones.add(*zones)

            if settings.SEND_TRANSACTIONAL_EMAILS:
                invited_by_name = invited_by.first_name
                SendGridClient().send_email_with_template(
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    to_emails=email,
                    subject=f"{invited_by_name} invited you to join their team on Voxel!",
                    template_id=USER_INVITATION_TEMPLATE_ID,
                    invited_by_name=invited_by_name,
                    organization_name=organization.name,
                    register_url=f"{settings.BASE_URL}/register/{token}",
                )
