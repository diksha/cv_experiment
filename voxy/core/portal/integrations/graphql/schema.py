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

from core.portal.api.models.incident import Incident
from core.portal.api.models.organization import Organization
from core.portal.devices.models.camera import Camera
from core.portal.incidents.graphql.types import FilterInputType, IncidentType
from core.portal.zones.models.zone import Zone


class IntegrationFields(graphene.ObjectType):

    filtered_raw_incidents = DjangoConnectionField(
        IncidentType,
        from_utc=graphene.DateTime(),
        to_utc=graphene.DateTime(),
        organization_key=graphene.List(graphene.String),
        zone_key=graphene.List(graphene.String),
        camera_uuid=graphene.List(graphene.String),
        incident_type_filter=graphene.String(),
        feedback_type=graphene.String(),
    )

    raw_incidents_from_filters = DjangoConnectionField(
        IncidentType,
        from_utc=graphene.DateTime(),
        to_utc=graphene.DateTime(),
        filters=graphene.List(FilterInputType),
    )

    def resolve_filtered_raw_incidents(
        self,
        info: graphene.ResolveInfo,
        from_utc: Optional[datetime],
        to_utc: Optional[datetime],
        organization_key: str,
        zone_key: str,
        camera_uuid: str,
        incident_type_filter: str,
        feedback_type: str,
        **_: None,
    ) -> QuerySet[Incident]:
        if not info.context.user.is_superuser:
            raise RuntimeError("Operation allowed only by super user")
        queryset = (
            Incident.objects_raw.from_timestamp(from_utc)
            .to_timestamp(to_utc)
            .exclude(corrupt_feedback_count__gt=0)
            .exclude(unsure_feedback_count__gt=0)
        )
        incident_type_q = Q(
            incident_type__key__iexact=incident_type_filter
        ) | Q(data__incident_type_id__iexact=incident_type_filter)
        queryset = queryset.filter(incident_type_q)

        # Filter by validity
        if feedback_type == "invalid":
            queryset = queryset.filter(invalid_feedback_count__gt=0,).exclude(
                valid_feedback_count__gt=0,
            )
        elif feedback_type == "valid":
            queryset = queryset.filter(valid_feedback_count__gt=0,).exclude(
                invalid_feedback_count__gt=0,
            )

        # Filter by organization/zone/camera
        if organization_key:
            queryset = queryset.filter(
                organization__in=Organization.objects.filter(
                    key__in=organization_key
                )
            )

        if zone_key:
            queryset = queryset.filter(
                zone__in=Zone.objects.filter(key__in=zone_key)
            )

        if camera_uuid:
            queryset = queryset.filter(
                camera__in=Camera.objects.filter(uuid__in=camera_uuid)
            )

        return queryset

    def resolve_raw_incidents_from_filters(
        self,
        info: graphene.ResolveInfo,
        from_utc: Optional[datetime],
        to_utc: Optional[datetime],
        filters: Optional[List[FilterInputType]] = None,
        **_: None,
    ) -> QuerySet[Incident]:
        """
        Resolves all incidents with a set of filters defined in

        core/portal/api/models/incident_filters

        Args:
            info (graphene.ResolveInfo): resolver info
            from_utc (Optional[datetime]): start date in iso format
            to_utc (Optional[datetime]): end date in iso format
            filters (Optional[List[FilterInputType]], optional): A list of filters to apply.
                See core/portal/incidents/enums for which filters to apply. Defaults to None.
            **_ (None): ignore

        Raises:
            RuntimeError: if this is not executed by a super user

        Returns:
            QuerySet[Incident]: the filtered query set
        """
        if not info.context.user.is_superuser:
            raise RuntimeError("Operation allowed only by super user")
        queryset = (
            Incident.objects_raw.from_timestamp(from_utc)
            .to_timestamp(to_utc)
            .apply_filters(
                FilterInputType.to_filter_list(filters), info.context.user
            )
        )
        return queryset


class IntegrationsQueries(graphene.ObjectType):
    """Queries for internal use only. Do not expose these to clients."""

    integrations = graphene.Field(IntegrationFields)

    def resolve_integrations(
        self,
        info: graphene.ResolveInfo,
    ) -> IntegrationFields:
        return IntegrationFields()
