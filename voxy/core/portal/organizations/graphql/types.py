#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
from datetime import date
from typing import List, Optional

import graphene
from django.contrib.auth.models import User
from django.db.models.query import QuerySet
from graphene_django import DjangoConnectionField, DjangoObjectType

from core.portal.accounts.graphql.types import Role as RoleType
from core.portal.accounts.graphql.types import UserType
from core.portal.accounts.models.role import Role as RoleModel
from core.portal.activity.graphql.types import SessionCount
from core.portal.activity.services import (
    get_organization_user_session_counts,
    get_organization_weekly_average_site_session_counts,
)
from core.portal.api.models.incident_type import IncidentType
from core.portal.api.models.organization import (
    Organization as OrganizationModel,
)
from core.portal.demos.data.constants import DEMO_SCORE_DATA
from core.portal.incidents.graphql.types import OrganizationIncidentTypeType
from core.portal.lib.utils.score_utils import (
    handle_if_event_score_name_organization_override,
)
from core.portal.scores.graphql.types import Score
from core.portal.scores.services import (
    calculate_all_organizational_event_scores,
)
from core.portal.zones.graphql.types import ZoneType
from core.portal.zones.models import Zone


class OrganizationType(DjangoObjectType):
    class Meta:
        model = OrganizationModel
        interfaces = [graphene.relay.Node]
        fields: List[str] = [
            "id",
            "pk",
            "name",
            "key",
            "users",
            "timezone",
            "is_sandbox",
        ]
        filter_fields: List[str] = []

    pk = graphene.Int()
    users = DjangoConnectionField(UserType)
    sites = graphene.List(ZoneType)
    incident_types = graphene.List(
        graphene.NonNull(OrganizationIncidentTypeType), required=True
    )
    roles = graphene.List(graphene.NonNull(RoleType))
    overall_score = graphene.Field(
        Score,
        start_date=graphene.Date(
            required=True,
            description="Start date, inclusive",
        ),
        end_date=graphene.Date(
            required=True,
            description="End date, inclusive",
        ),
    )
    event_scores = graphene.Field(
        graphene.List(Score),
        start_date=graphene.Date(
            required=True,
            description="Start date, inclusive",
        ),
        end_date=graphene.Date(
            required=True,
            description="End date, inclusive",
        ),
    )
    session_count = graphene.Field(
        graphene.NonNull(SessionCount),
        start_date=graphene.Date(
            required=True,
            description="Start date, inclusive",
        ),
        end_date=graphene.Date(
            required=True,
            description="End date, inclusive",
        ),
    )

    @staticmethod
    def resolve_overall_score(
        parent: "OrganizationType",
        info: graphene.ResolveInfo,
        start_date: date,
        end_date: date,
    ) -> Optional[Score]:
        """Resolve organization score

        Args:
            parent (OrganizationType): parent object
            info (graphene.ResolveInfo): graphene context
            start_date (date): start of event date filter range
            end_date (date): end of event date filter range

        Returns:
            Score: A score object
        """
        demo_data = DEMO_SCORE_DATA["organizations"].get(parent.key)
        if demo_data is not None:
            return demo_data["overallScore"]

        results = calculate_all_organizational_event_scores(
            organization_ids=[parent.id],
            start_date=start_date,
            end_date=end_date,
        )
        total = 0
        num_event_scores = 0
        for result in results:
            if (
                result.get("site_id") is None
                and result.get("organization_id") == parent.id
            ):
                total += result.get("calculated_score")
                num_event_scores += 1

        if num_event_scores == 0:
            return None

        score_value = total / num_event_scores
        return Score(value=int(score_value), label=parent.name)

    @staticmethod
    def resolve_event_scores(
        parent: "OrganizationType",
        info: graphene.ResolveInfo,
        start_date: date,
        end_date: date,
    ) -> list[Score]:
        """Resolve event type scores

        Args:
            parent (OrganizationType): parent object
            info (graphene.ResolveInfo): graphene context
            start_date (date): start of event date filter range
            end_date (date): end of event date filter range

        Returns:
            list[Score]: A list of incident type score object
        """
        demo_data = DEMO_SCORE_DATA["organizations"].get(parent.key)
        if demo_data is not None:
            return demo_data["eventScores"]

        results = calculate_all_organizational_event_scores(
            organization_ids=[parent.id],
            start_date=start_date,
            end_date=end_date,
        )
        event_scores = []
        for result in results:
            if (
                result.get("site_id") is None
                and result.get("organization_id") == parent.id
            ):
                event_scores.append(
                    Score(
                        label=result.get("score_name"),
                        value=result.get("calculated_score"),
                    )
                )

        handle_if_event_score_name_organization_override(
            organization_key=parent.key, event_scores=event_scores
        )

        return event_scores

    @staticmethod
    def resolve_session_count(
        parent: "OrganizationType",
        info: graphene.ResolveInfo,
        start_date: date,
        end_date: date,
    ) -> SessionCount:
        """Resolve session count

        Args:
            parent (OrganizationType): parent object
            info (graphene.ResolveInfo): graphene context
            start_date (date): start of event date filter range
            end_date (date): end of event date filter range

        Returns:
            SessionCount: A SessionCount object
        """
        return SessionCount(
            sites=get_organization_weekly_average_site_session_counts(
                parent,
                start_date,
                end_date,
            ),
            users=get_organization_user_session_counts(
                parent,
                start_date,
                end_date,
            ),
        )

    def resolve_users(
        self, _: graphene.ResolveInfo, **__: None
    ) -> QuerySet[User]:
        return (
            # Exclude service/bot accounts
            self.users.filter(email__isnull=False)
            .exclude(email__exact="")
            .order_by("-date_joined")
        )

    def resolve_incident_types(
        self, _: graphene.ResolveInfo
    ) -> QuerySet[IncidentType]:
        return self.enabled_incident_types

    def resolve_sites(
        self: OrganizationModel, _: graphene.ResolveInfo
    ) -> QuerySet[ZoneType]:
        # TODO: role based access control
        return Zone.objects.filter(organization=self, parent_zone__isnull=True)

    def resolve_roles(
        self: OrganizationModel, _: graphene.ResolveInfo
    ) -> QuerySet[RoleType]:
        return RoleModel.objects.filter(visible_to_customers=True)
