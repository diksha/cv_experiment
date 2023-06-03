import typing as t
from datetime import datetime

from django.contrib import admin
from django.contrib.auth.models import User
from django.db import models
from django.db.models import Q

from core.portal.activity.constants import DELIMITER
from core.portal.activity.enums import SessionScope
from core.portal.api.models.organization import Organization
from core.portal.lib.models.base import Model
from core.portal.zones.models.zone import Zone


class UserSession(Model):
    """User session model"""

    class Meta:
        """Meta class"""

        verbose_name = "User Session"
        verbose_name_plural = "User Sessions"
        ordering = ["-created_at"]
        app_label = "activity"
        db_table = "user_session"

        constraints = [
            models.CheckConstraint(
                name="allow_site_or_org_or_neither_but_not_both_check",
                check=(
                    models.Q(site__isnull=True, organization__isnull=False)
                    | models.Q(site__isnull=False, organization__isnull=True)
                    | models.Q(site__isnull=True, organization__isnull=True)
                ),
            ),
            # We need 3 separate unique constraints here because the organization
            # and site fields are nullable, Postgres does not consider
            # NULL equal to NULL, and we only ever want one of org or site to
            # contain a value (see check constraint above). So if we used a single
            # unique constraint across user/org/site/start_timestamp we would never
            # violate the constraint because there will always be at least one NULL
            # value. Instead, we need unique constraints with predicates to enforce
            # uniqueness for each of the 3 possible scopes.
            models.UniqueConstraint(
                fields=[
                    "user",
                    "start_timestamp",
                ],
                name="unique_session_global_scope_by_start_timestamp",
                condition=Q(organization__isnull=True, site__isnull=True),
            ),
            models.UniqueConstraint(
                fields=[
                    "user",
                    "organization",
                    "start_timestamp",
                ],
                name="unique_session_org_scope_by_start_timestamp",
                condition=Q(site__isnull=True),
            ),
            models.UniqueConstraint(
                fields=[
                    "user",
                    "site",
                    "start_timestamp",
                ],
                name="unique_session_site_scope_by_start_timestamp",
                condition=Q(organization__isnull=True),
            ),
        ]

    # This key field exists because we want to efficiently bulk upsert
    # into this table and we can't (easily) use the composite unique
    # constraints defined above for that purpose. This is because Django
    # uses Postgres' ON CONFLICT syntax for bulk_create, but the
    # ON CONFLICT args must match a unique index on the table,
    # including any predicates, and Django does not provide a way
    # for us to specify a predicate as part of bulk_create.
    #
    # So rather than fall back to raw SQL, we "denormalize" the
    # unique values into a string key with a unique constraint and
    # use this field as the unique field in bulk_create, and it works
    # fine because there is no predicate.
    #
    # See this post for more details:
    # https://betakuang.medium.com/why-postgresqls-on-conflict-cannot-find-my-partial-unique-index-552327b85e1
    key = models.CharField(
        null=False,
        blank=False,
        unique=True,
        max_length=100,
    )

    start_timestamp = models.DateTimeField(null=False, blank=False)
    end_timestamp = models.DateTimeField(null=False, blank=False)

    user = models.ForeignKey(
        User,
        null=False,
        blank=False,
        on_delete=models.CASCADE,
        related_name="user_sessions",
    )

    site = models.ForeignKey(
        Zone,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="user_sessions",
    )
    organization = models.ForeignKey(
        Organization,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="user_sessions",
    )

    def __str__(self) -> str:
        """String representation of the user session model"""
        return f"{self.user} - {self.site} - {self.organization}"

    @classmethod
    def build_key(
        cls,
        user_id: int,
        scope: SessionScope,
        scope_id: t.Union[int, str],
        start_timestamp: datetime,
    ) -> str:
        """Build a unique session key.

        Args:
            user_id (int): user id
            scope (SessionScope): session scope
            scope_id (t.Union[int, str]): session scope id
            start_timestamp (datetime): session start timestamp

        Returns:
            str: unique session key
        """
        return DELIMITER.join(
            [
                str(user_id),
                str(scope),
                str(scope_id),
                str(int(start_timestamp.timestamp())),
            ]
        )


admin.site.register(UserSession)
