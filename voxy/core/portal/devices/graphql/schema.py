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

import graphene
from django.db.models.query import QuerySet

from core.portal.api.models.organization import Organization
from core.portal.devices.graphql.mutations import (
    CameraConfigNewCreate,
    CameraCreate,
    CameraUpdate,
    EdgeCreate,
)
from core.portal.devices.graphql.types import (
    CameraConfigNewModelType,
    CameraType,
    EdgeType,
)
from core.portal.devices.models.camera import Camera as CameraModel
from core.portal.devices.models.camera import (
    CameraConfigNew as CameraConfigNewModel,
)
from core.portal.devices.models.edge import Edge as EdgeModel

# TODO: remove this in favor of RBAC check
ORG_KEY_ALLOWLIST = ["AMERICOLD", "USCOLD", "HANSEN"]


class CameraConfigNewMutations(graphene.ObjectType):
    camera_config_new_create = CameraConfigNewCreate.Field()


class CameraMutations(graphene.ObjectType):
    camera_create = CameraCreate.Field()
    camera_update = CameraUpdate.Field()


class CameraQueries(graphene.ObjectType):

    cameras = graphene.List(CameraType)
    camera_config_new = graphene.Field(
        CameraConfigNewModelType,
        uuid=graphene.String(),
        version=graphene.Int(),
    )

    def resolve_cameras(
        self, info: graphene.ResolveInfo
    ) -> QuerySet[CameraType]:
        """Resolve cameras

        Args:
            info: graphene.ResolveInfo

        Returns:
            QuerySet[CameraType]: queryset of camera type
        """
        current_org = info.context.user.profile.current_organization
        if not current_org:
            return CameraModel.objects.none()

        if current_org.key in ORG_KEY_ALLOWLIST:
            return CameraModel.objects.filter(
                organization=Organization.objects.get(key=current_org.key)
            )
        return CameraModel.objects.all()

    def resolve_camera_config_new(
        self, info: graphene.ResolveInfo, uuid: str, version: int
    ) -> CameraConfigNewModel:
        """Resolve camera config

        Args:
            info (graphene.ResolveInfo): graphene.ResolveInfo
            uuid (str): camera uuid
            version (int): config version

        Returns:
            CameraConfigNewModel: camera config model
        """
        camera = CameraModel.objects.get(uuid=uuid)
        return CameraConfigNewModel.objects.get(camera=camera, version=version)


class EdgeQueries(graphene.ObjectType):

    edges = graphene.List(EdgeType)

    def resolve_edges(self, info: graphene.ResolveInfo) -> QuerySet[EdgeType]:
        """Reoslve edges

        Args:
            info (graphene.ResolveInfo): graphene.ResolveInfo

        Returns:
            QuerySet[EdgeType]: queryset of edge type
        """
        current_org = info.context.user.profile.current_organization
        if not current_org:
            return EdgeModel.objects.none()

        return EdgeModel.objects.all()


class EdgeMutations(graphene.ObjectType):

    edge_create = EdgeCreate.Field()
