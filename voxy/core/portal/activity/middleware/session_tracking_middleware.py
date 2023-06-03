import time
import typing as t
from dataclasses import dataclass
from functools import cached_property

from django.contrib.auth.models import AnonymousUser, User
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse
from loguru import logger

from core.portal.activity.constants import (
    ACTIVE_SESSION_KEY_PREFIX,
    DELIMITER,
    ETL_SESSION_KEY_PREFIX,
    GLOBAL_SESSION_SCOPE_ID,
    SESSION_TIMEOUT_SECONDS,
)
from core.portal.activity.enums import SessionScope


class SessionTrackingMiddleware:
    """Middleware to track user sessions.

    We maintain two cache entries per user per session scope:

    Active entries:
        - Used to track active sessions
        - Expire based on session timeout threshold via Redis TTLs
        - As new activity occurs, timestamps and TTLS get refreshed
        - Format:
            - key: user_session:active:$USER_ID:$SESSION_SCOPE:$SCOPE_ID
            - value: $START_TIMESTAMP:$END_TIMESTAMP
        - Example:
            {
                'user_session:active:980:1:GLOBAL': '1682618205:1682618307',
                'user_session:active:980:2:20955': '1682618205:1682618307',
                'user_session:active:980:3:269': '1682618205:1682618307'
            }

    ETL entries:
        - Used to track both active and recently ended sessions
        - Consumed by external ETL process
        - Timestamp values get updated as activity occurs
        - No TTLs set, entries are read, inserted into DB, then deleted
          via ETL process
        - Format:
            - key: user_session:etl:$USER_ID:$SESSION_SCOPE:$SCOPE_ID:$START_TIMESTAMP
            - value: $END_TIMESTAMP
        - Example:
            {
                'user_session:etl:980:1:GLOBAL:1682618205': 1682618307,
                'user_session:etl:980:2:20955:1682618205': 1682618307,
                'user_session:etl:980:3:269:1682618205': 1682618307
            }

    TODO: should we exclude reviewers, internal users?
    TODO: get org/site from client request headers instead of backend profile
    """

    def __init__(self, get_response: t.Callable) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        current_user = request.user
        is_anonymous = isinstance(current_user, AnonymousUser)

        if current_user and not is_anonymous:
            try:
                self.track_session(current_user)
            # trunk-ignore(pylint/W0718): catch-all, prefer to lose tracking vs. fail the request
            except Exception:
                logger.exception("Error tracking user session")

        return self.get_response(request)

    def track_session(self, current_user: User) -> None:
        """Track user session.

        Args:
            current_user (User): current user
        """
        profile = current_user.profile
        current_org_id = profile.organization_id if profile else None
        current_site_id = profile.site_id if profile else None
        current_timestamp = int(time.time())
        key_builder = SessionKeyBuilder(
            current_user.id,
            current_org_id,
            current_site_id,
        )

        # If there are no valid active keys, return
        if not key_builder.all_active_keys:
            return

        # Get all active session entries
        active_session_entries = cache.get_many(key_builder.all_active_keys)

        # Dicts to collect entries to upsert
        active_session_entries_to_upsert = {}
        etl_session_entries_to_upsert = {}

        # Update global session entries
        if key_builder.global_active_key:
            global_active_value = active_session_entries.get(
                key_builder.global_active_key
            )
            global_start = self._get_start_timestamp_from_active_session_value(
                global_active_value,
                current_timestamp,
            )
            active_session_entries_to_upsert[
                key_builder.global_active_key
            ] = self._build_active_session_value(
                global_start,
                current_timestamp,
            )
            global_etl_key = key_builder.build_global_etl_key(global_start)
            etl_session_entries_to_upsert[global_etl_key] = current_timestamp

        # Update org session entries
        if key_builder.org_active_key:
            org_active_value = active_session_entries.get(
                key_builder.org_active_key
            )
            org_start = self._get_start_timestamp_from_active_session_value(
                org_active_value,
                current_timestamp,
            )
            active_session_entries_to_upsert[
                key_builder.org_active_key
            ] = self._build_active_session_value(
                org_start,
                current_timestamp,
            )
            org_etl_key = key_builder.build_org_etl_key(org_start)
            etl_session_entries_to_upsert[org_etl_key] = current_timestamp

        # Update site session entries
        if key_builder.site_active_key:
            site_active_value = active_session_entries.get(
                key_builder.site_active_key
            )
            site_start = self._get_start_timestamp_from_active_session_value(
                site_active_value,
                current_timestamp,
            )
            active_session_entries_to_upsert[
                key_builder.site_active_key
            ] = self._build_active_session_value(
                site_start,
                current_timestamp,
            )
            site_etl_key = key_builder.build_site_etl_key(site_start)
            etl_session_entries_to_upsert[site_etl_key] = current_timestamp

        # Upsert active session entries with TTL
        if bool(active_session_entries_to_upsert):
            cache.set_many(
                active_session_entries_to_upsert,
                timeout=SESSION_TIMEOUT_SECONDS,
            )

        # Upsert ETL session entries without TTL
        if bool(etl_session_entries_to_upsert):
            cache.set_many(
                etl_session_entries_to_upsert,
                # Specifying None is required to prevent default TTL of 300
                timeout=None,
            )

    def _get_start_timestamp_from_active_session_value(
        self,
        session_value: t.Optional[str],
        default_value: int,
    ) -> t.Tuple[int, int]:
        if session_value:
            try:
                start_timestamp_str, _ = session_value.split(DELIMITER)
                start_timestamp = int(start_timestamp_str)
                return start_timestamp
            except (ValueError, TypeError):
                logger.warning(f"Invalid session value: {session_value}")
        return default_value

    def _build_active_session_value(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> str:
        return DELIMITER.join([str(start_timestamp), str(end_timestamp)])


@dataclass
class SessionKeyBuilder:
    user_id: int
    organization_id: t.Optional[int] = None
    site_id: t.Optional[int] = None

    @cached_property
    def all_active_keys(self) -> t.List[str]:
        """All active session keys.

        Returns:
            t.List[str]: list of keys
        """
        keys = [
            self.global_active_key,
            self.org_active_key,
            self.site_active_key,
        ]
        return [key for key in keys if key is not None]

    @cached_property
    def global_active_key(self) -> str:
        """Global active key.

        Returns:
            str: global active key
        """
        return self._build_active_key(
            SessionScope.GLOBAL, GLOBAL_SESSION_SCOPE_ID
        )

    @cached_property
    def org_active_key(self) -> t.Optional[str]:
        """Organization active key.

        Returns:
            t.Optional[str]: organization active key
        """
        if self.organization_id:
            return self._build_active_key(
                SessionScope.ORGANIZATION, self.organization_id
            )
        return None

    @cached_property
    def site_active_key(self) -> t.Optional[str]:
        """Site active key.

        Returns:
            t.Optional[str]: site active key
        """
        if self.site_id:
            return self._build_active_key(SessionScope.SITE, self.site_id)
        return None

    def build_global_etl_key(self, start_timestamp: int) -> str:
        """Build global ETL key.

        Args:
            start_timestamp (int): session start timestamp

        Returns:
            str: global ETL key
        """
        return self._build_etl_key(
            SessionScope.GLOBAL,
            GLOBAL_SESSION_SCOPE_ID,
            start_timestamp,
        )

    def build_org_etl_key(self, start_timestamp: int) -> t.Optional[str]:
        """Build organization ETL key.

        Args:
            start_timestamp (int): session start timestamp

        Returns:
            t.Optional[str]: organization ETL key
        """
        if self.organization_id:
            return self._build_etl_key(
                SessionScope.ORGANIZATION,
                self.organization_id,
                start_timestamp,
            )
        return None

    def build_site_etl_key(self, start_timestamp: int) -> t.Optional[str]:
        """Build site ETL key.

        Args:
            start_timestamp (int): session start timestamp

        Returns:
            t.Optional[str]: site ETL key
        """
        if self.site_id:
            return self._build_etl_key(
                SessionScope.SITE,
                self.site_id,
                start_timestamp,
            )
        return None

    def _build_active_key(
        self,
        session_scope: SessionScope,
        scope_id: t.Union[int, str],
    ) -> str:
        return DELIMITER.join(
            [
                ACTIVE_SESSION_KEY_PREFIX,
                str(self.user_id),
                str(session_scope.value),
                str(scope_id),
            ]
        )

    def _build_etl_key(
        self,
        session_scope: SessionScope,
        scope_id: t.Union[int, str],
        start_timestamp: t.Union[int, str],
    ) -> str:
        return DELIMITER.join(
            [
                ETL_SESSION_KEY_PREFIX,
                str(self.user_id),
                str(session_scope.value),
                str(scope_id),
                str(start_timestamp),
            ]
        )
