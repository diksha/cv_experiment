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
from datetime import datetime
from typing import List, Optional

import graphene
from django.db.models import Q
from django.db.models.query import QuerySet
from graphene_django import DjangoConnectionField
from loguru import logger

from core.portal.accounts.permissions import (
    EXPERIMENTAL_INCIDENTS_READ,
    INCIDENT_DETAILS_READ,
)
from core.portal.accounts.permissions_manager import has_zone_permission
from core.portal.api.models.incident import Incident
from core.portal.api.models.incident import IncidentType as IncidentTypeModel
from core.portal.incidents.graphql.mutations import (
    CreateUserIncident,
    DeleteUserIncident,
    IncidentCreateScenario,
    IncidentCreateShareLink,
    IncidentExportVideo,
    IncidentHighlight,
    IncidentReopen,
    IncidentResolve,
    IncidentUndoHighlight,
)
from core.portal.incidents.graphql.types import (
    FilterInputType,
    IncidentType,
    IncidentTypeType,
)
from core.portal.lib.graphql.exceptions import PermissionDenied
from core.portal.lib.graphql.utils import pk_from_global_id


class IncidentTypeCount(graphene.ObjectType):
    """Incident type count."""

    incident_type = graphene.Field(IncidentTypeType)

    count = graphene.Int()


class IncidentTypeQueries(graphene.ObjectType):

    incident_types = graphene.List(IncidentTypeType, required=True)

    def resolve_incident_types(
        self, info: graphene.ResolveInfo
    ) -> "QuerySet[IncidentTypeModel]":
        return IncidentTypeModel.objects.all()


class IncidentQueries(graphene.ObjectType):
    incident_feed = DjangoConnectionField(
        IncidentType,
        from_utc=graphene.DateTime(),
        to_utc=graphene.DateTime(),
        filters=graphene.List(FilterInputType),
        priority_filter=graphene.List(graphene.String),
        status_filter=graphene.List(graphene.String),
        incident_type_filter=graphene.List(graphene.String),
        camera_filter=graphene.List(graphene.String),
        list_filter=graphene.List(graphene.String),
        assignee_filter=graphene.List(graphene.String),
        experimental_filter=graphene.Boolean(),
    )

    incident_details = graphene.Field(
        IncidentType,
        incident_id=graphene.ID(),
        incident_uuid=graphene.String(),
    )

    @staticmethod
    def apply_filters_to_queryset(
        info: graphene.ResolveInfo,
        from_utc: Optional[datetime],
        to_utc: Optional[datetime],
        filters: Optional[List[FilterInputType]],
        incident_type_filter: Optional[List[str]] = None,
        priority_filter: Optional[List[str]] = None,
        status_filter: Optional[List[str]] = None,
        camera_filter: Optional[List[str]] = None,
        list_filter: Optional[List[str]] = None,
        assignee_filter: Optional[List[str]] = None,
        experimental_filter: Optional[bool] = False,
        **_: None,
    ) -> QuerySet[Incident]:
        base_queryset = Incident.objects.for_user(info.context.user)

        # Allow superusers to query experimental incidents
        if experimental_filter:
            if info.context.user.is_superuser:
                base_queryset = Incident.objects_experimental
            else:
                base_queryset = Incident.objects.none()

        queryset = (
            base_queryset.with_bookmarked_flag(info.context.user)
            .from_timestamp(from_utc)
            .to_timestamp(to_utc)
            .apply_filters(
                FilterInputType.to_filter_list(filters), info.context.user
            )
        )

        # TODO: remove in favor of filters argument
        # Filter by priority
        if priority_filter and "all" not in priority_filter:
            priority_q = Q()
            for priority in priority_filter or []:
                priority_q = priority_q | Q(priority__iexact=priority)
            queryset = queryset.filter(priority_q)

        # TODO: remove in favor of filters argument
        # Filter by status
        if status_filter and "all" not in status_filter:
            status_q = Q()
            for status in status_filter or []:
                status_q = status_q | Q(status__iexact=status)
            queryset = queryset.filter(status_q)

        # TODO: remove in favor of filters argument
        # Filter by incident type
        if incident_type_filter and "all" not in incident_type_filter:
            incident_type_q = Q()
            for incident_type in incident_type_filter or []:
                incident_type_q = (
                    incident_type_q
                    | Q(incident_type__key__iexact=incident_type)
                    | Q(data__incident_type_id__iexact=incident_type)
                )
            queryset = queryset.filter(incident_type_q)

        # TODO: remove in favor of filters argument
        # Filter by camera
        if camera_filter and "all" not in camera_filter:
            camera_q = Q()
            for camera in camera_filter or []:
                camera_q = camera_q | Q(data__camera_uuid__iexact=camera)
            queryset = queryset.filter(camera_q)

        # TODO: remove in favor of filters argument
        # Filter by list
        if list_filter:
            queryset = queryset.filter(
                lists__id__in=list_filter,
                lists__owner=info.context.user,
            )

        # TODO: remove in favor of filters argument
        # Filter by assignee
        if assignee_filter and "all" not in assignee_filter:
            assignee_q = Q()
            for assignee_id in assignee_filter or []:
                _, pk = pk_from_global_id(assignee_id)
                assignee_q = assignee_q | Q(user_incidents__assignee__id=pk)
            queryset = queryset.filter(assignee_q)

        return queryset

    # trunk-ignore-all(pylint/R0912): too many branches
    def resolve_incident_feed(
        self,
        info: graphene.ResolveInfo,
        from_utc: Optional[datetime] = None,
        to_utc: Optional[datetime] = None,
        filters: Optional[List[FilterInputType]] = None,
        incident_type_filter: Optional[List[str]] = None,
        priority_filter: Optional[List[str]] = None,
        status_filter: Optional[List[str]] = None,
        camera_filter: Optional[List[str]] = None,
        list_filter: Optional[List[str]] = None,
        assignee_filter: Optional[List[str]] = None,
        experimental_filter: Optional[bool] = False,
        **_: None,
    ) -> QuerySet[Incident]:
        """Returns a QuerySet which can be paginated by the caller."""

        queryset = IncidentQueries.apply_filters_to_queryset(
            info,
            from_utc,
            to_utc,
            filters,
            incident_type_filter,
            priority_filter,
            status_filter,
            camera_filter,
            list_filter,
            assignee_filter,
            experimental_filter,
        )
        return queryset.order_by("-timestamp")

    def resolve_incident_details(
        self,
        info: graphene.ResolveInfo,
        incident_id: Optional[str] = None,
        incident_uuid: Optional[str] = None,
    ) -> IncidentType:

        if incident_id and incident_uuid:
            raise ValueError(
                "Only one of incident_id or incident_uuid are allowed but"
                " both values were provided."
            )

        if not incident_id and not incident_uuid:
            raise ValueError(
                "One of incident_id or incident_uuid are required."
            )

        if incident_id:
            _, incident_pk = pk_from_global_id(incident_id)
            filter_q = Q(pk=incident_pk)
        else:
            filter_q = Q(uuid=incident_uuid)

        matching_incidents = Incident.objects_raw.filter(
            filter_q
        ).with_bookmarked_flag(info.context.user)

        if matching_incidents.count() == 1:
            incident = matching_incidents.first()
        elif matching_incidents.count() > 1:
            logger.error(f"Multiple incidents match filters: {filter_q}")

        if incident:
            if incident.experimental:
                if has_zone_permission(
                    info.context.user,
                    incident.zone,
                    EXPERIMENTAL_INCIDENTS_READ,
                ):
                    return incident
            elif has_zone_permission(
                info.context.user, incident.zone, INCIDENT_DETAILS_READ
            ):
                return incident

        raise PermissionDenied(
            "You do not have permission to view incident details."
        )


class IncidentMutations(graphene.ObjectType):

    assign_incident = CreateUserIncident.Field()
    unassign_incident = DeleteUserIncident.Field()
    incident_create_scenario = IncidentCreateScenario.Field()
    incident_resolve = IncidentResolve.Field()
    incident_reopen = IncidentReopen.Field()
    incident_export_video = IncidentExportVideo.Field()
    incident_highlight = IncidentHighlight.Field()
    incident_undo_highlight = IncidentUndoHighlight.Field()
    incident_create_share_link = IncidentCreateShareLink.Field()
