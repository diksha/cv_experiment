import uuid

from core.portal.api.models.organization import Organization
from core.portal.zones.models.zone import Zone


def backfill_anonymous_keys():
    """Backfill anonymous keys"""

    # First backfill organizations
    organizations = Organization.objects.filter(anonymous_key__isnull=True)
    for organization in organizations:
        organization.anonymous_key = uuid.uuid4()
        organization.save()

    # Then backfill zones
    zones = Zone.objects.filter(anonymous_key__isnull=True)
    for zone in zones:
        zone.anonymous_key = uuid.uuid4()
        zone.save()


if __name__ == "__main__":
    backfill_anonymous_keys()
