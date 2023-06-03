from typing import List

import graphene
from graphene_django import DjangoObjectType

from core.portal.api.models.incident import Incident
from core.portal.api.models.incident_feedback import IncidentFeedback


class IncidentFeedbackType(DjangoObjectType):
    class Meta:
        model = IncidentFeedback
        fields = "__all__"


class FeedbackComment(graphene.ObjectType):
    feedback_text = graphene.String()
    user_email = graphene.String()


class IncidentFeedbackSummaryType(DjangoObjectType):
    class Meta:
        model = Incident
        interfaces = [graphene.relay.Node]
        exclude = [
            "deleted_at",
            "highlighted",
            "cooldown_source",
            "review_status",
        ]
        # TODO: utilize django-filter instead of custom *_filter arguments
        filter_fields: List[str] = []

    pk = graphene.Int()
    valid = graphene.List(graphene.String, required=True)
    invalid = graphene.List(graphene.String, required=True)
    unsure = graphene.List(graphene.String, required=True)


class ReviewQueueContext(graphene.InputObjectType):
    review_panel_id = graphene.Int()
    incident_exclusion_list = graphene.List(graphene.String)
