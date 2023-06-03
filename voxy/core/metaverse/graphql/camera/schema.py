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
from copy import deepcopy

import graphene

from core.metaverse.graphql.model.schema import TaskSchema
from core.metaverse.models.camera import Camera


class CameraConfigPolygonSchema(graphene.ObjectType):
    polygon = graphene.List(graphene.List(graphene.Float))


class PointSchema(graphene.ObjectType):
    points = graphene.List(graphene.List(graphene.Float))


class DoorSchema(graphene.ObjectType):
    polygon = graphene.List(graphene.List(graphene.Float))
    orientation = graphene.String()
    door_id = graphene.Int()


class CameraConfigSchema(graphene.ObjectType):
    uuid = graphene.String()
    doors = graphene.List(DoorSchema)
    driving_areas = graphene.List(CameraConfigPolygonSchema)
    actionable_regions = graphene.List(CameraConfigPolygonSchema)
    intersections = graphene.List(CameraConfigPolygonSchema)
    end_of_aisles = graphene.List(PointSchema)
    no_pedestrian_zones = graphene.List(CameraConfigPolygonSchema)
    motion_detection_zones = graphene.List(CameraConfigPolygonSchema)
    no_obstruction_regions = graphene.List(CameraConfigPolygonSchema)


class CameraSchema(graphene.ObjectType):
    uuid = graphene.String()
    created_timestamp = graphene.String()
    organization = graphene.String()
    location = graphene.String()
    zone = graphene.String()
    channel_name = graphene.String()
    kinesis_url = graphene.String()
    is_active = graphene.Boolean()
    camera_config_ref = graphene.List(CameraConfigSchema)
    task_ref = graphene.List(TaskSchema)


class CameraQueries(graphene.ObjectType):
    """Queries for camera node in metaverse

    Args:
        graphene: Type of object
    """

    camera = graphene.List(
        CameraSchema,
        uuid=graphene.String(),
        organization=graphene.String(),
        location=graphene.String(),
        zone=graphene.String(),
        channel_name=graphene.String(),
    )

    cameras_for_task = graphene.List(CameraSchema, task_uuid=graphene.String())

    def resolve_cameras_for_task(self, _, *__, **kwargs) -> list:
        """Given task get all cameras

        Args:
            kwargs:
                task_uuid: uuid of the task
            __: unused

        Returns:
            list: List of cameras associated with a task.
        """
        return [
            camera
            for (camera, task) in [
                (camera, task)
                for camera in Camera.nodes.all()
                for task in camera.task_ref.all()
            ]
            if task.uuid == kwargs.get("task_uuid")
        ]

    def resolve_camera(
        self, _, *__, **kwargs
    ) -> Camera:  # pylint: disable=unused-argument
        """Returns camera given uuid or other location information.

        Args:
            kwargs:
                uuid: uuid of the camera
                organization: Org of the camera
                location: Location of the camera
                zone: Zone of the camera
                channel: Channel of the camera
            __: unused


        Returns:
            Camera: Camera corresponding to arguments
        """
        params = deepcopy(kwargs)
        if kwargs.get("uuid"):
            params.pop("uuid")
            return Camera().nodes.filter(uuid=kwargs["uuid"])
        return Camera().nodes.filter(**params)
