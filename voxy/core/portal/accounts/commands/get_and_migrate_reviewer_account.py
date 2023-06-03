import typing as t

from django.contrib.auth.models import User
from django.db.models import Q

from core.portal.lib.commands import CommandABC


class GetAndMigrateReviewerAccount(CommandABC):
    """Command to get and migrate a reviewer account.

    This is a temporary step to migrate existing reviewer accounts to use
    the new Uber SSO Auth0 connection.
    """

    def __init__(self, auth0_id: str, email: str) -> None:
        self.auth0_id = auth0_id
        self.email = email

    def execute(self) -> t.Optional[User]:
        """Execute the command.

        Returns:
            t.Optional[User]: user instance if found, else None
        """
        # Get the user instance via either auth0_id or email
        user = (
            User.objects.filter(
                Q(profile__data__auth0_id=self.auth0_id)
                | Q(email__iexact=self.email)
            )
            .prefetch_related("profile")
            .first()
        )

        # If no user matches, that's fine, the auth backend will handle it
        if not user:
            return None

        # Ensure Auth0 ID matches user profile, this is the "migration" part
        if user.profile.data.get("auth0_id") != self.auth0_id:
            user.profile.data["auth0_id"] = self.auth0_id
            user.profile.save()
        return user
