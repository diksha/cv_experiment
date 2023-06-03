from random import SystemRandom
from typing import List, Optional

from django.contrib.auth.models import User
from django.db.models import Count, F

from core.portal.api.models.incident import Incident
from core.portal.incident_feedback.lock import (
    get_locked_incident_ids,
    lock_incident,
)
from core.portal.incidents.enums import IncidentTypeKey

COOLDOWN_REVIEW_THRESHOLD = 20


def get_incident_to_review(
    current_user: User,
    excluded_incident_uuids: List[str],
) -> Optional[Incident]:
    """Get an incident which is eligible for review.

    Args:
        current_user (User): current user
        excluded_incident_uuids (List[str]): list of incident UUIDs to exclude

    Returns:
        Optional[Incident]: incident to review (or None if no incidents are eligible for review)
    """
    incident_to_review = None
    use_lock = True
    profile_data = current_user.profile.data or {}

    # Get priority org key list and ensure it's a list
    priority_org_keys = profile_data.get("review_queue_priority_org_keys")
    priority_org_keys = (
        priority_org_keys if isinstance(priority_org_keys, list) else None
    )

    # Get priority site key list and ensure it's a list
    priority_site_keys = profile_data.get("review_queue_priority_site_keys")
    priority_site_keys = (
        priority_site_keys if isinstance(priority_site_keys, list) else None
    )

    # If the current user has priority orgs or sites specified,
    # try to get an incident from those sites WITHOUT locking it
    # to prioritize low latency over minimizing duplicate reviews
    if priority_org_keys or priority_site_keys:
        incident_to_review = _get_incident_to_review_with_filters(
            current_user,
            excluded_incident_uuids,
            include_sandbox_orgs=True,
            include_experimental_incidents=True,
            include_cooldown_incidents=True,
            org_key_filter=priority_org_keys,
            site_key_filter=priority_site_keys,
            use_lock=False,
        )

        if incident_to_review:
            use_lock = False

    # If no incidents was fetched for user-defined priority
    # get a high-priority incident from high priority sites
    if not incident_to_review:
        incident_to_review = _get_incident_to_review_with_filters(
            current_user,
            excluded_incident_uuids,
            include_cooldown_incidents=True,
            high_priority_sites_only=True,
        )

    # If no incident was fetched for a priority site,
    # get a non-cooldown incident without any filters
    if not incident_to_review:
        queue_counts = Incident.reviewable_objects.filter(
            cooldown_source__isnull=True
        ).aggregate(non_cooldown_incident_count=Count("pk"))

        should_review_cooldown_incidents = (
            queue_counts.get("non_cooldown_incident_count", 0)
            < COOLDOWN_REVIEW_THRESHOLD
        )

        incident_to_review = _get_incident_to_review_with_filters(
            current_user,
            excluded_incident_uuids,
            include_cooldown_incidents=should_review_cooldown_incidents,
        )

    # If there are no non-cooldown incidents that need review,
    # get a cooldown incident
    if not incident_to_review:
        incident_to_review = _get_incident_to_review_with_filters(
            current_user,
            excluded_incident_uuids,
            include_cooldown_incidents=True,
        )

    if incident_to_review and use_lock:
        lock_incident(incident_to_review.id)

    return incident_to_review


# trunk-ignore(pylint/R0912): too many args fine for now
# trunk-ignore(pylint/R0913): too many branches fine for now
def _get_incident_to_review_with_filters(
    current_user: User,
    excluded_incident_uuids: List[str],
    include_sandbox_orgs: bool = False,
    include_experimental_incidents: bool = False,
    include_cooldown_incidents: bool = False,
    org_key_filter: Optional[List[str]] = None,
    site_key_filter: Optional[List[str]] = None,
    high_priority_sites_only: bool = False,
    use_lock: bool = True,
) -> Optional[Incident]:
    """Get an incident which is eligible for review (with filters applied).

    Args:
        current_user (User): current user
        excluded_incident_uuids (List[str]): list of incident UUIDs to exclude
        include_sandbox_orgs (bool): true if query should include incidents
            from sandbox organizations, otherwise false. By default we exclude
            sandbox data from the review queue, but we may want to override this
            for certain experimental use cases.
        include_experimental_incidents (bool): true if query should include
            experimental incidents
        include_cooldown_incidents (bool): true if query should include
            cooldown incidents
        org_key_filter (Optional[List[str]]): org key filter
        site_key_filter (Optional[List[str]]): site key filter
        high_priority_sites_only (bool): true if query should include only high priority sites
        use_lock (bool): true if query should lock the incident


    Returns:
        Optional[Incident]: incident to review (or None if no incidents are eligible for review)
    """

    base_queryset = Incident.reviewable_objects.all()

    if not include_experimental_incidents:
        base_queryset = base_queryset.filter(experimental=False)

    excluded_incident_ids = get_locked_incident_ids() if use_lock else []

    # This queryset represents all incidents which are
    # eligible for the review queue for the current user,
    filtered_queryset = (
        base_queryset.exclude(feedback__user=current_user)
        # Exclude any incidents currently locked by another reviewer
        .exclude(id__in=excluded_incident_ids)
        # Exclude any incidents currently buffered by the requester
        .exclude(uuid__in=excluded_incident_uuids).order_by("-timestamp")
    )

    if is_uber_email(current_user):
        # Exclude certain incident types for Uber reviewers
        filtered_queryset = filtered_queryset.exclude(
            incident_type__key__in=[
                IncidentTypeKey.OVERREACHING_WITH_SAFETY_UNIFORM,
                IncidentTypeKey.BAD_POSTURE_WITH_SAFETY_UNIFORM,
                IncidentTypeKey.N_PERSON_PED_ZONE,
                IncidentTypeKey.BUMP_CAP,
                IncidentTypeKey.DOOR_VIOLATION,
            ],
        )

    if not include_sandbox_orgs:
        # Exclude demo/sandbox organizations
        filtered_queryset = filtered_queryset.exclude(
            organization__is_sandbox=True
        )

    if high_priority_sites_only:
        filtered_queryset = filtered_queryset.filter(
            zone__is_high_priority=True
        )

    if org_key_filter:
        filtered_queryset = filtered_queryset.filter(
            organization__key__in=org_key_filter
        )

    if site_key_filter:
        filtered_queryset = filtered_queryset.filter(
            zone__key__in=site_key_filter
        )

    if not include_cooldown_incidents:
        filtered_queryset = filtered_queryset.filter(
            cooldown_source__isnull=True
        )

    # Order matters here. First we query by camera so that we don't starve a customer,
    # then we query by incident_type.
    camera_pool = (
        filtered_queryset.values_list("camera", flat=True)
        .distinct()
        .order_by()
    )

    if camera_pool:
        filtered_queryset = filtered_queryset.filter(
            camera=SystemRandom().choice(list(camera_pool))
        )

    incident_type_id_pool = (
        filtered_queryset.values_list("incident_type_id", flat=True)
        .distinct()
        .order_by()
    )

    if incident_type_id_pool:
        filtered_queryset = filtered_queryset.filter(
            incident_type_id=SystemRandom().choice(list(incident_type_id_pool))
        )

    # Fetch a batch of incidents
    incident_id_pool = (
        filtered_queryset.values_list("id", flat=True)
        .distinct()
        .order_by()[:25]
    )

    # check cache again
    excluded_incident_ids = get_locked_incident_ids() if use_lock else []
    incident_id_pool = [
        id for id in incident_id_pool if id not in excluded_incident_ids
    ]

    # Pick a random incident
    if incident_id_pool:
        incident_ids = list(incident_id_pool)

        while incident_ids:
            selected_incident_id = SystemRandom().choice(incident_ids)
            incident = Incident.objects_raw.get(id=selected_incident_id)

            if not incident:
                continue

            if use_lock:
                if lock_incident(incident.id):
                    return incident
            else:
                return incident

            incident_ids.remove(selected_incident_id)

    return None


def get_incident_to_review_for_shadow_reviewers(
    current_user: User,
    excluded_incident_uuids: List[str],
) -> Optional[Incident]:
    """Get an incident which is eligible for shadow review.

    Args:
        current_user (User): current user (shadow reviewer)
        excluded_incident_uuids (List[str]): list of incident UUIDs to exclude

    Returns:
        Optional[Incident]: incident to review (or None if no incidents eligible)
    """

    excluded_incident_ids = get_locked_incident_ids()

    # Get the incidents that are non-cooldown, have at least 1 feedback from production review,
    # and are of specific incident types.
    base_queryset = (
        Incident.objects.annotate(
            feedback_count=F("valid_feedback_count")
            + F("invalid_feedback_count")
        )
        .filter(
            cooldown_source__isnull=True,
            feedback_count__gte=1,
            data__shadow_reviewed__isnull=True,
        )
        .exclude(id__in=excluded_incident_ids)
        .exclude(uuid__in=excluded_incident_uuids)
    )

    # exclude these incident keys
    base_queryset = base_queryset.exclude(
        incident_type__key__in=[
            IncidentTypeKey.OVERREACHING_WITH_SAFETY_UNIFORM,
            IncidentTypeKey.BAD_POSTURE_WITH_SAFETY_UNIFORM,
            IncidentTypeKey.N_PERSON_PED_ZONE,
            IncidentTypeKey.BUMP_CAP,
            IncidentTypeKey.DOOR_VIOLATION,
        ],
    )

    base_queryset = base_queryset.order_by("-timestamp")
    # Fetch a batch of incidents
    incident_id_pool = base_queryset.values_list("id", flat=True).distinct()[
        0:50
    ]

    # check cache again
    excluded_incident_ids = get_locked_incident_ids()
    incident_id_pool = [
        id for id in incident_id_pool if id not in excluded_incident_ids
    ]

    # Pick a random incident
    if incident_id_pool:
        incident_ids = list(incident_id_pool)

        while incident_ids:
            selected_incident_id = SystemRandom().choice(incident_ids)
            incident = Incident.objects_raw.get(id=selected_incident_id)

            if incident and lock_incident(incident.id):
                return incident

            incident_ids.remove(selected_incident_id)

    return None


def is_uber_email(user: User) -> bool:
    """Check if a user has an Uber email address.

    Args:
        user (User): The user.

    Returns:
        bool: True if the user has an Uber email address, False otherwise.
    """
    return user.email.endswith("@uber.com") or user.email.endswith(
        "@ext.uber.com"
    )
