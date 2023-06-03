from datetime import datetime, timezone

from django.core.cache import cache
from loguru import logger

from core.portal.activity.constants import DELIMITER, ETL_SESSION_KEY_WILDCARD
from core.portal.activity.enums import SessionScope
from core.portal.activity.models import UserSession
from core.portal.lib.jobs.base import JobBase


class SessionTrackingETLJob(JobBase):
    def run(self) -> None:
        """Move user session data from cache to db.

        This is a somewhat sensitive operation, without a transaction
        mechanism if this process fails partway there is risk we will
        lose session data. I think we're ok with this risk for now,
        but it's worth noting.

        We delete the ETL entries immediately after reading them to
        reduce likelyhood of race conditions. It's still possible
        that an ETL entry will get updated between the time we read
        the entry and delete it, though, which would make the end
        timestamp of the session incorrect. This is another risk
        I think we're ok with for now.
        """

        # Pop session ETL entries from cache
        etl_keys = cache.keys(ETL_SESSION_KEY_WILDCARD)
        etl_entries = cache.get_many(etl_keys)
        cache.delete_many(etl_keys)

        # Build session objects
        user_sessions = []
        for key, value in etl_entries.items():
            key_parts = key.split(DELIMITER)
            if not len(key_parts) == 6:
                # Log and skip invalid keys
                logger.warning(f"Invalid session ETL key: {key}")
                continue

            _, _, user_id, scope_enum, scope_id, start_timestamp = key_parts
            end_timestamp = value

            try:
                # Convert key parts to correct types
                user_id = int(user_id)
                scope = SessionScope(int(scope_enum))
                start_timestamp = datetime.fromtimestamp(
                    int(start_timestamp), tz=timezone.utc
                )
                end_timestamp = datetime.fromtimestamp(
                    int(end_timestamp), tz=timezone.utc
                )

                # Convert scope id to correct type (or None for global scope)
                org_id = (
                    int(scope_id)
                    if scope == SessionScope.ORGANIZATION
                    else None
                )
                site_id = int(scope_id) if scope == SessionScope.SITE else None

                session_key = UserSession.build_key(
                    user_id,
                    scope,
                    scope_id,
                    start_timestamp,
                )

                user_sessions.append(
                    UserSession(
                        key=session_key,
                        user_id=user_id,
                        organization_id=org_id,
                        site_id=site_id,
                        start_timestamp=start_timestamp,
                        end_timestamp=end_timestamp,
                    )
                )
            except (ValueError, TypeError):
                # Log and skip invalid entries
                logger.warning(
                    f"Invalid session ETL entry: key={key}, value={value}"
                )
                continue

        # Bulk upsert
        UserSession.objects.bulk_create(
            user_sessions,
            update_conflicts=True,
            unique_fields=["key"],
            update_fields=["end_timestamp"],
        )

        logger.info(f"Upserted {len(user_sessions)} user session entries")
