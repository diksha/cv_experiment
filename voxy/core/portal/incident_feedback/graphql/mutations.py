from typing import Optional

import graphene

from core.portal.accounts.graphql.types import UserError
from core.portal.accounts.permissions import INCIDENT_FEEDBACK_CREATE
from core.portal.accounts.permissions_manager import has_zone_permission
from core.portal.api.models.incident import Incident
from core.portal.incident_feedback.graphql.types import IncidentFeedbackType
from core.portal.incident_feedback.services import (
    CreateIncidentFeedbackService,
    CreateShadowIncidentFeedbackService,
)
from core.portal.lib.graphql.utils import pk_from_global_id


class CreateIncidentFeedback(graphene.Mutation):
    class Arguments:
        incident_id = graphene.ID(required=True)
        feedback_type = graphene.String(required=True)
        feedback_value = graphene.String(required=True)
        feedback_text = graphene.String(required=True)
        elapsed_milliseconds_between_reviews = graphene.Int()
        incident_served_timestamp_seconds = graphene.Int()

    incident_feedback = graphene.Field(IncidentFeedbackType)
    user_errors = graphene.List(UserError)

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        incident_id: str,
        feedback_type: str,
        feedback_value: str,
        feedback_text: str,
        elapsed_milliseconds_between_reviews: Optional[int] = None,
        incident_served_timestamp_seconds: Optional[int] = None,
    ) -> "CreateIncidentFeedback":
        """Creates an incident feedback and returns it.

        Args:
            root (None): root object
            info (graphene.ResolveInfo): resolveInfo obj
            incident_id (str):  ID of the incident
            feedback_type (str): type of feedback
            feedback_value (str): value of feedback
            feedback_text (str): feedback comment
            elapsed_milliseconds_between_reviews (int): time spent on review (in ms)
            incident_served_timestamp_seconds (int): time the incident was served (in seconds)
        Returns:
            CreateIncidentFeedback: instance of feedback
        """
        del root, feedback_type

        _, incident_pk = pk_from_global_id(incident_id)
        incident = Incident.objects_raw.get(pk=incident_pk)

        if not has_zone_permission(
            info.context.user,
            incident.zone,
            INCIDENT_FEEDBACK_CREATE,
        ):
            user_error = UserError(
                message="You do not have permission to create incident feedback.",
            )
            return CreateIncidentFeedback(user_errors=[user_error])

        if info.context.user.profile.data.get("shadow_reviewer", False):
            CreateShadowIncidentFeedbackService(
                info.context.user,
                incident,
                feedback_value,
                feedback_text,
                elapsed_milliseconds_between_reviews,
                incident_served_timestamp_seconds,
            ).execute()
            return CreateIncidentFeedback()

        feedback = CreateIncidentFeedbackService(
            info.context.user,
            incident,
            feedback_value,
            feedback_text,
            elapsed_milliseconds_between_reviews,
            incident_served_timestamp_seconds,
        ).execute()

        return CreateIncidentFeedback(incident_feedback=feedback)
