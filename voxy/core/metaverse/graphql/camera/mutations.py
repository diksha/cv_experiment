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

from core.metaverse.graphql.camera.schema import (
    CameraConfigSchema,
    CameraSchema,
)
from core.metaverse.models.camera import Camera, CameraConfig


class CameraConfigPolygonArgument(graphene.InputObjectType):
    polygon = graphene.List(graphene.List(graphene.Float))


class PointArgument(graphene.InputObjectType):
    points = graphene.List(graphene.List(graphene.Float))


class DoorArgument(graphene.InputObjectType):
    polygon = graphene.List(graphene.List(graphene.Float))
    orientation = graphene.String()
    door_id = graphene.Int()


class CameraCreate(graphene.Mutation):
    class Arguments:
        organization = graphene.String(required=True)
        location = graphene.String(required=True)
        zone = graphene.String(required=True)
        channel_name = graphene.String(required=True)
        kinesis_url = graphene.String()
        is_active = graphene.Boolean()
        uuid = graphene.String()

    success = graphene.Boolean()
    camera = graphene.Field(CameraSchema)

    def mutate(self, info, **kwargs):  # pylint: disable=unused-argument
        camera = Camera(**kwargs)
        camera.save()

        return CameraCreate(camera=camera, success=True)


class CameraMutations(graphene.ObjectType):
    camera_create = CameraCreate.Field()


class CameraConfigCreate(graphene.Mutation):
    class Arguments:
        organization = graphene.String(required=True)
        location = graphene.String(required=True)
        zone = graphene.String(required=True)
        channel_name = graphene.String(required=True)
        version = graphene.String(required=True)
        doors = graphene.List(DoorArgument)
        driving_areas = graphene.List(CameraConfigPolygonArgument)
        actionable_regions = graphene.List(CameraConfigPolygonArgument)
        intersections = graphene.List(CameraConfigPolygonArgument)
        end_of_aisles = graphene.List(PointArgument)
        no_pedestrian_zones = graphene.List(CameraConfigPolygonArgument)
        motion_detection_zones = graphene.List(CameraConfigPolygonArgument)
        no_obstruction_regions = graphene.List(CameraConfigPolygonArgument)

    success = graphene.Boolean()
    camera_config = graphene.Field(CameraConfigSchema)

    def mutate(self, info, **kwargs):  # pylint: disable=unused-argument
        camera_ref = Camera.nodes.get(
            organization=kwargs["organization"],
            location=kwargs["location"],
            zone=kwargs["zone"],
            channel_name=kwargs["channel_name"],
        )
        # Delete previous camera_config. We will implement versioning in the next cls.
        for camera_config in camera_ref.camera_config_ref:
            camera_config.delete()
        camera_config = CameraConfig(**kwargs).save()
        camera_ref.camera_config_ref.connect(camera_config)
        return CameraConfigCreate(camera_config=camera_config, success=True)


class CameraConfigMutations(graphene.ObjectType):
    camera_config_create = CameraConfigCreate.Field()
