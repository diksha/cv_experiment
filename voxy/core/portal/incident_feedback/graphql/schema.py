from datetime import datetime
from typing import List, Optional

import graphene
from django.contrib.postgres.aggregates import ArrayAgg
from django.db.models import Count, Q
from django.db.models.query import QuerySet
from graphene_django import DjangoConnectionField

from core.portal.accounts.permissions import (
    INCIDENT_FEEDBACK_READ,
    REVIEW_QUEUE_READ,
)
from core.portal.accounts.permissions_manager import has_global_permission
from core.portal.api.models.incident_feedback import Incident
from core.portal.incident_feedback.graphql.mutations import (
    CreateIncidentFeedback,
)
from core.portal.incident_feedback.graphql.types import (
    IncidentFeedbackSummaryType,
    ReviewQueueContext,
)
from core.portal.incident_feedback.services import (
    get_incident_to_review,
    get_incident_to_review_for_shadow_reviewers,
)
from core.portal.incidents.graphql.types import IncidentType
from core.portal.lib.graphql.exceptions import PermissionDenied
from core.portal.lib.graphql.utils import pk_from_global_id


class IncidentFeedbackQueries(graphene.ObjectType):

    incident_feedback_summary = DjangoConnectionField(
        IncidentFeedbackSummaryType,
        internal_feedback=graphene.String(),
        external_feedback=graphene.String(),
        has_comments=graphene.Boolean(),
        incident_type=graphene.String(),
        organization_id=graphene.String(),
        site_id=graphene.String(),
        from_utc=graphene.DateTime(),
        to_utc=graphene.DateTime(),
    )
    incident_feedback_queue = graphene.List(
        IncidentType,
        review_queue_context=graphene.Argument(
            ReviewQueueContext, required=True
        ),
        required=True,
    )

    def resolve_incident_feedback_summary(
        self,
        info: graphene.ResolveInfo,
        first: Optional[int] = None,
        after: Optional[int] = None,
        internal_feedback: Optional[str] = None,
        external_feedback: Optional[str] = None,
        has_comments: Optional[bool] = None,
        incident_type: Optional[str] = None,
        organization_id: Optional[str] = None,
        site_id: Optional[str] = None,
        from_utc: Optional[datetime] = None,
        to_utc: Optional[datetime] = None,
    ) -> QuerySet[IncidentFeedbackSummaryType]:
        # unused pagination variables, pagination is handled by graphene at runtime
        del first, after

        if not has_global_permission(
            info.context.user, INCIDENT_FEEDBACK_READ
        ):
            raise PermissionDenied(
                "You do not have permission to view incident feedback."
            )

        internal_filter = Q(feedback__user__email__contains="@voxel")

        def feedback_count(feedback_value: str, internal: bool) -> Count:
            filter_q = Q(feedback__feedback_value=feedback_value) & (
                internal_filter if internal else ~internal_filter
            )
            return Count("feedback", filter=filter_q)

        def comment_count(internal: bool) -> Count:
            filter_q = (
                Q(feedback__feedback_text__isnull=False)
                & ~Q(feedback__feedback_text__exact="")
                & (internal_filter if internal else ~internal_filter)
            )
            return Count("feedback", filter=filter_q)

        incidents_with_feedback = Incident.objects.filter(
            last_feedback_submission_timestamp__isnull=False
        )

        incidents_with_feedback = incidents_with_feedback.annotate(
            valid_internal_count=feedback_count("valid", True),
            valid_external_count=feedback_count("valid", False),
            invalid_internal_count=feedback_count("invalid", True),
            invalid_external_count=feedback_count("invalid", False),
            unsure_internal_count=feedback_count("unsure", True),
            unsure_external_count=feedback_count("unsure", False),
            internal_comment_count=comment_count(True),
            external_comment_count=comment_count(False),
        )

        def filter_feedback_type(feedback_type: str, internal: bool) -> Q:
            filters_map = {
                "valid": Q(valid_internal_count__gt=0)
                if internal
                else Q(valid_external_count__gt=0),
                "invalid": Q(invalid_internal_count__gt=0)
                if internal
                else Q(invalid_external_count__gt=0),
                "unsure": Q(unsure_internal_count__gt=0)
                if internal
                else Q(unsure_external_count__gt=0),
            }
            return filters_map[feedback_type]

        filters_q = Q()
        if internal_feedback and internal_feedback != "all":
            filters_q = filters_q & filter_feedback_type(
                internal_feedback, True
            )
        if external_feedback and external_feedback != "all":
            filters_q = filters_q & filter_feedback_type(
                external_feedback, False
            )
        if has_comments is True:
            filters_q = filters_q & (
                Q(internal_comment_count__gt=0)
                | Q(external_comment_count__gt=0)
            )
        if incident_type and incident_type != "all":
            filters_q = filters_q & Q(incident_type__key=incident_type)
        if organization_id and organization_id != "all":
            _, organization_pk = pk_from_global_id(organization_id)
            filters_q = filters_q & Q(organization_id=organization_pk)
        if site_id and site_id != "all":
            _, site_pk = pk_from_global_id(site_id)
            filters_q = filters_q & Q(zone_id=site_pk)
        incidents_with_feedback = (
            incidents_with_feedback.filter(filters_q)
            .from_last_feedback_submission_timestamp(from_utc)
            .to_last_feedback_submission_timestamp(to_utc)
            .annotate(
                valid=ArrayAgg(
                    "feedback__user__email",
                    filter=Q(feedback__feedback_value="valid"),
                ),
                invalid=ArrayAgg(
                    "feedback__user__email",
                    filter=Q(feedback__feedback_value="invalid"),
                ),
                unsure=ArrayAgg(
                    "feedback__user__email",
                    filter=Q(feedback__feedback_value="unsure"),
                ),
            )
            .order_by("-last_feedback_submission_timestamp")
        )
        return incidents_with_feedback

    def resolve_incident_feedback_queue(
        self,
        info: graphene.ResolveInfo,
        review_queue_context: ReviewQueueContext,
    ) -> List[Incident]:
        if not has_global_permission(info.context.user, REVIEW_QUEUE_READ):
            raise PermissionDenied(
                "You do not have permission to view the incident feedback queue."
            )

        if info.context.user.profile.data.get("shadow_reviewer", False):
            incident_to_review = get_incident_to_review_for_shadow_reviewers(
                info.context.user,
                excluded_incident_uuids=review_queue_context.incident_exclusion_list
                or [],
            )
            return [incident_to_review] if incident_to_review else []

        incident_to_review = get_incident_to_review(
            info.context.user,
            excluded_incident_uuids=review_queue_context.incident_exclusion_list
            or [],
        )

        if incident_to_review:
            return [incident_to_review]
        return []


class IncidentFeedbackMutations(graphene.ObjectType):
    create_incident_feedback = CreateIncidentFeedback.Field()
