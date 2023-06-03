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
import copy
from typing import Any

import graphene
from deepdiff import DeepDiff
from django.core.exceptions import ObjectDoesNotExist
from django.db.models import Max
from django.forms.models import model_to_dict

from core.portal.accounts.permissions import CAMERAS_RENAME
from core.portal.accounts.permissions_manager import has_zone_permission
from core.portal.api.models.organization import Organization
from core.portal.devices.graphql.types import (
    CameraConfigNewModelType,
    CameraType,
    EdgeType,
)
from core.portal.devices.models.camera import Camera as CameraModel
from core.portal.devices.models.camera import (
    CameraConfigNew as CameraConfigModel,
)
from core.portal.devices.models.edge import Edge as EdgeModel
from core.portal.lib.graphql.mutations import BaseMutation
from core.portal.lib.graphql.utils import pk_from_global_id
from core.utils.type_check import is_valid_uuid


class CameraConfigNewCreate(BaseMutation):
    class Arguments:
        uuid = graphene.String(required=True)
        doors = graphene.JSONString()
        driving_areas = graphene.JSONString()
        actionable_regions = graphene.JSONString()
        intersections = graphene.JSONString()
        end_of_aisles = graphene.JSONString()
        no_pedestrian_zones = graphene.JSONString()
        motion_detection_zones = graphene.JSONString()
        no_obstruction_regions = graphene.JSONString()

    camera_config_new = graphene.Field(CameraConfigNewModelType)
    is_updated = graphene.Boolean()

    @staticmethod
    def mutate(
        root: None, info: graphene.ResolveInfo, uuid: str, **kwargs: Any
    ) -> "CameraConfigNewCreate":
        """
        Args:
            root: root
            info: graphene.ResolveInfo
            uuid(str): camera UUID
            **kwargs: camera config params

        Raises:
            RuntimeError: no uuid

        Returns:
            CameraConfigNewCreate: camera config
        """
        del root
        if not uuid:
            raise RuntimeError("UUID is required")
        camera = CameraModel.objects.get(uuid=uuid)
        try:
            camera_config_new = CameraConfigModel.objects.get(
                camera=camera,
                version=CameraConfigModel.objects.filter(
                    camera=camera
                ).aggregate(Max("version"))["version__max"],
            )

        except ObjectDoesNotExist:
            # Camera does not exist
            camera_config_new = None
        if not camera_config_new:
            camera_config_new = CameraConfigModel(camera=camera)
        camera_old = copy.deepcopy(camera_config_new)
        is_updated = False
        camera_config_new.doors = kwargs.get("doors", [])
        camera_config_new.driving_areas = kwargs.get("driving_areas", [])
        camera_config_new.actionable_regions = kwargs.get(
            "actionable_regions", []
        )
        camera_config_new.intersections = kwargs.get("intersections", [])
        camera_config_new.end_of_aisles = kwargs.get("end_of_aisles", [])
        camera_config_new.no_pedestrian_zones = kwargs.get(
            "no_pedestrian_zones", []
        )
        camera_config_new.motion_detection_zones = kwargs.get(
            "motion_detection_zones", []
        )
        camera_config_new.no_obstruction_regions = kwargs.get(
            "no_obstruction_regions", []
        )
        if DeepDiff(
            model_to_dict(camera_config_new), model_to_dict(camera_old)
        ):
            if not camera_old.version:
                camera_config_new.version = 1
            else:
                diction = model_to_dict(camera_config_new)
                diction["version"] = camera_old.version + 1
                del diction["id"]
                del diction["deleted_at"]
                del diction["camera"]
                camera_new = CameraConfigModel(**diction, camera=camera)
                camera_new.version = camera_old.version + 1
                camera_config_new = camera_new
            is_updated = True
            camera_config_new.save()
        return CameraConfigNewCreate(
            camera_config_new=camera_config_new, is_updated=is_updated
        )


class CameraCreate(BaseMutation):
    class Arguments:
        """CameraCreate Arguments
        Properties:
            organization_key (str): organization key
            zone_key (str): zone key
            camera_uuid (str): camera uuid
            camera_name (str): camera name
        """

        organization_key = graphene.String(required=True)
        zone_key = graphene.String(required=True)
        camera_uuid = graphene.String(required=True)
        camera_name = graphene.String(required=True)

    camera = graphene.Field(CameraType)

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        organization_key: str,
        zone_key: str,
        camera_uuid: str,
        camera_name: str,
    ) -> "CameraCreate":
        """
        Create a camera given a set of keyword arguments.

        Arguments:
            root: root
            info: graphene.ResolveInfo
            organization_key (str): organization key
            zone_key (str): zone key
            camera_uuid (str): camera uuid
            camera_name (str): camera name
        Raises:
            RuntimeError: Camera with UUID alread exists

        Returns:
            CameraCreate: camera create
        """
        # Only superusers can create cameras

        if not info.context.user.is_superuser:
            raise RuntimeError("User not authorized for creating cameras")

        organization = Organization.objects.get(key=organization_key)
        zone = organization.sites.get(key=zone_key)

        # Camera must not have duplicate uuid
        if CameraModel.objects.filter(uuid=camera_uuid).exists():
            raise RuntimeError(f"Camera UUID already exists: {camera_uuid}")

        camera_instance = CameraModel.objects.create(
            uuid=camera_uuid,
            name=camera_name,
            zone=zone,
            organization=organization,
        )
        return CameraCreate(camera=camera_instance)


class CameraUpdate(BaseMutation):
    """Update an camera identified by its id"""

    class Arguments:
        """
        The arguments for the mutation
        Properties:
            camera_id (str): camera id
            camera_name (str): camera name
        """

        camera_id = graphene.ID(required=True)
        camera_name = graphene.String(required=True)

    camera = graphene.Field(CameraType)

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        camera_id: str,
        camera_name: str,
    ) -> "CameraUpdate":
        """The mutate function
        Arguments:
            root: root
            info: graphene.ResolveInfo
            camera_id (str): camera id
            camera_name (str): camera name
        Raises:
            PermissionError: User not authorized for updating cameras
            DoesNotExist: Model does not exist

        Returns:
            CameraUpdate: camera update
        """
        _, camera_pk = pk_from_global_id(camera_id)
        camera = CameraModel.objects.get(pk=camera_pk)
        if not has_zone_permission(
            info.context.user,
            camera.zone,
            CAMERAS_RENAME,
        ):
            return PermissionError("User not authorized for updating camera")

        if camera_name:
            camera.name = camera_name
        camera.save()
        return CameraUpdate(camera=camera)


class EdgeUpdate(BaseMutation):
    """Update an edge identified by its uuid"""

    class Arguments:
        """
        The arguments for the mutation
        Properties:
            uuid (str): the uuid of the edge to update
            lifecycle (str): the lifecycle of the edge
            description (str): the description of the edge
            mac_address (str): the mac address of the edge
            serial (str): the serial of the edge
        """

        uuid = graphene.String(required=True)

        # optional fields
        lifecycle = graphene.String(
            description="The lifecycle stage of the edge device",
        )
        description = graphene.String(description="Description of the edge")
        mac_address = graphene.String(
            description="The MAC address of the edge device"
        )
        serial = graphene.String(
            description="The serial number of the edge device"
        )

    edge = graphene.Field(EdgeType)

    @staticmethod
    # trunk-ignore(pylint/W9016)
    def mutate(
        root=None,
        uuid=str,
        lifecycle=str,
        description=str,
        mac_address=str,
        serial=str,
    ) -> "EdgeUpdate":
        """
        Update an edge device

        Args:
            root: root
            uuid: the uuid of the edge device
            lifecycle: the lifecycle stage of the edge device
            description: description of the edge
            mac_address: the MAC address of the edge device
            serial: the serial number of the edge device

        Raises:
            RuntimeError: Invalid UUID

        Returns:
            EdgeUpdate: edge update
        """

        if not is_valid_uuid(uuid):
            raise RuntimeError("Invalid UUID")
        edge = EdgeModel.objects.get(uuid=uuid)
        if lifecycle:
            edge.lifecycle = lifecycle
        if description:
            edge.description = description
        if mac_address:
            edge.mac_address = mac_address
        if serial:
            edge.serial = serial

        edge.save()
        return EdgeUpdate(edge=edge)


class EdgeCreate(BaseMutation):
    """Create an edge device
    A UUID will automatically be generated for the edge
    The status will be set to a default value of "OFFLINE"
    """

    class Arguments:
        """The arguments for the mutation"""

        # required fields
        name = graphene.String(
            description="The name of the edge device", required=True
        )
        organization_key = graphene.String(
            description="The organization key", required=True
        )
        name = graphene.String(
            description="The name of the edge", required=True
        )

        # optional fields
        description = graphene.String(description="Description of the edge")
        mac_address = graphene.String(
            description="The MAC address of the edge device"
        )
        serial = graphene.String(
            description="The serial number of the edge device"
        )

    # edge = graphene.Field(EdgeType)

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        organization_key: str,
        name: str,
        description: str,
        mac_address: str,
        serial: str,
    ) -> "EdgeCreate":
        """Create an edge device

        Args:
            root: root
            info: graphene.ResolveInfo
            organization_key: organization key of the edge
            name: name of the edge
            description: description of the edge
            mac_address: the MAC address of the edge device
            serial: the serial number of the edge device

        Raises:
            RuntimeError: User not authorized for creating cameras
            RuntimeError: Edge name is required
            RuntimeError: Organization key is required

        Returns:
            EdgeCreate: edge create
        """

        # Null not checks
        if not info.context.user.is_superuser:
            raise RuntimeError("User not authorized for creating cameras")
        if not name:
            raise RuntimeError("Edge name is required")
        if not organization_key:
            raise RuntimeError("Organization key is required")

        # Validation
        organization_key = Organization.objects.get(key=organization_key)

        instance = EdgeModel.objects.create(
            name=name,
            organization=organization_key,
            description=description,
            mac_address=mac_address,
            serial=serial,
        )

        return EdgeCreate(edge=instance)
