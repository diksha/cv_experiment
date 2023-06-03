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
from typing import Any

import graphene
from graphene.types.generic import GenericScalar
from loguru import logger
from neomodel import DoesNotExist, NeomodelException, db

from core.metaverse.graphql.datacollection.schema import DataCollectionSchema
from core.metaverse.models.camera import Camera
from core.metaverse.models.datacollection import (
    Actor,
    DataCollection,
    Frame,
    LabelMetadata,
    VersionedViolation,
)
from core.structs.data_collection import DataCollectionType


class ActivityArgument(graphene.InputObjectType):
    activity = graphene.String()
    posture = graphene.String()


class VertexArgument(graphene.InputObjectType):
    x = graphene.Float()
    y = graphene.Float()
    z = graphene.Float()


class PolygonArgument(graphene.InputObjectType):
    vertices = graphene.List(VertexArgument)


class ActorArgument(graphene.InputObjectType):
    category = graphene.String()
    occluded = graphene.Boolean()
    occluded_degree = graphene.String()
    manual = graphene.Boolean()
    mirrored = graphene.Boolean()
    truncated = graphene.Boolean()
    human_operating = graphene.Boolean()
    forklift = graphene.Boolean()
    loaded = graphene.Boolean()
    forks_raised = graphene.Boolean()
    operating_pit = graphene.Boolean()
    operating_object = graphene.String()
    is_wearing_hard_hat = graphene.Boolean()
    motion_detection_zone_state = graphene.String()
    is_wearing_safety_vest = graphene.Boolean()
    is_wearing_safety_vest_v2 = graphene.Boolean()
    is_carrying_object = graphene.Boolean()
    door_state = graphene.String()
    door_type = graphene.String()
    door_orientation = graphene.String()
    track_id = graphene.Int()
    track_uuid = graphene.String()
    polygon = graphene.Field(PolygonArgument)
    pose = graphene.String()
    activity = GenericScalar()
    head_covering_type = graphene.String()


class FrameArgument(graphene.InputObjectType):
    frame_number = graphene.Int()
    frame_width = graphene.Int()
    frame_height = graphene.Int()
    relative_timestamp_s = graphene.Float()
    relative_timestamp_ms = graphene.Float()
    epoch_timestamp_ms = graphene.Float()
    path_of_image = graphene.String()
    actors = graphene.List(ActorArgument)
    relative_image_path = graphene.String()


class LabelMetadataArgument(graphene.InputObjectType):
    """Schema for label as an argument

    Args:
        graphene: graphql input object type
    """

    source = graphene.String()
    project_name = graphene.String()
    taxonomy = graphene.String()
    taxonomy_version = graphene.String()


class ViolationsArgument(graphene.InputObjectType):
    version = graphene.String()
    violations = graphene.List(graphene.String)


class DataCollectionArgument(graphene.InputObjectType):
    camera_uuid = graphene.String()
    name = graphene.String()
    path = graphene.String()
    is_test = graphene.Boolean()
    violations = graphene.List(ViolationsArgument)
    frames = graphene.List(FrameArgument)


class DataCollectionCreate(graphene.Mutation):
    class Arguments:
        camera_uuid = graphene.String()
        name = graphene.String()
        path = graphene.String()
        is_test = graphene.Boolean()
        voxel_uuid = graphene.String()
        violations = graphene.List(ViolationsArgument)
        frames = graphene.List(FrameArgument)
        label_metadata = graphene.Argument(LabelMetadataArgument)
        data_collection_type = graphene.String()

    success = graphene.Boolean()
    data_collection = graphene.Field(DataCollectionSchema)

    def mutate(
        self,
        info: graphene.ResolveInfo,
        camera_uuid: str,
        path: str,
        **kwargs: Any,
    ) -> "DataCollectionCreate":  # pylint: disable=unused-argument
        """
        Creates a data collection

        Args:
            info (graphene.ResolveInfo): the graphene resolver information
            camera_uuid (str): the current camera uuid
            path (str): the path of the data collection
            **kwargs (Any): the rest of the input arguments as a dict

        Raises:
           ValueError: if the data_collection_type is not passed
           ValueError: if the data_collection_type is not found in the original enum

        Returns:
            DataCollectionCreate: the data collection create object
        """
        # Fail query if datacollection exists
        data_collection = DataCollection.nodes.first_or_none(path=path)
        if data_collection:
            return DataCollectionCreate(data_collection=None, success=False)
        try:
            if not kwargs.get("data_collection_type"):
                raise ValueError("No data collection type found in mutation!")

            if not (
                kwargs.get("data_collection_type")
                in DataCollectionType.names()
            ):
                raise ValueError(
                    f"DataCollectionType: {kwargs.get('data_collection_type')} "
                    f"not found in available types: {DataCollectionType.names()}"
                )
            db.begin()
            data_collection = DataCollection(path=path, **kwargs).save()
            try:
                camera_ref = Camera.nodes.get(uuid=camera_uuid)
            except DoesNotExist:
                camera_info = camera_uuid.split("/")
                camera_ref = Camera(
                    uuid=camera_uuid,
                    organization=camera_info[0],
                    location=camera_info[1],
                    zone=camera_info[2],
                    channel_name=camera_info[3],
                    is_active=True,
                )
                camera_ref.save()
            data_collection.camera_ref.connect(camera_ref)
            if kwargs.get("violations") is not None:
                for violation in kwargs["violations"]:
                    violation_val = VersionedViolation(**violation).save()
                    data_collection.violation_ref.connect(violation_val)
            if kwargs.get("frames") is not None:
                for frame in kwargs["frames"]:
                    frame_val = Frame(**frame).save()
                    data_collection.frame_ref.connect(frame_val)
                    _add_actors(frame["actors"], frame_val)
            if kwargs.get("label_metadata") is not None:
                label = LabelMetadata(**(kwargs["label_metadata"])).save()
                data_collection.label_metadata_ref.connect(label)
            db.commit()
            success = True
        except Exception:  # trunk-ignore(pylint/W0703)
            db.rollback()
            logger.exception("Error encountered in DataCollectionCreate")
            success = False
        return DataCollectionCreate(
            data_collection=data_collection, success=success
        )


class DataCollectionUpdate(graphene.Mutation):
    class Arguments:
        voxel_uuid = graphene.String()
        violations = graphene.List(ViolationsArgument)
        frames = graphene.List(FrameArgument)
        label_metadata = graphene.Argument(LabelMetadataArgument)

    success = graphene.Boolean()
    data_collection = graphene.Field(DataCollectionSchema)

    def mutate(
        self,
        info: graphene.ResolveInfo,
        voxel_uuid: str,
        **kwargs: Any,
    ) -> "DataCollectionUpdate":  # pylint: disable=unused-argument
        """
        Creates a data collection

        Args:
            info (graphene.ResolveInfo): the graphene resolver information
            voxel_uuid (str): uuid of the data collection
            **kwargs (Any): the rest of the input arguments as a dict

        Raises:
            RuntimeError : Taxonomy projects are not the same

        Returns:
            DataCollectionUpdate: the data collection create object
        """
        try:
            try:
                db.begin()
                data_collection = DataCollection.nodes.first(
                    voxel_uuid=voxel_uuid
                )
                # If data_collection exists, delete frames, actors and create again.
                # trunk-ignore(pylint/C0301)
                delete_query = f'match (n:DataCollection{{uuid:"{data_collection.uuid}"}}) -->(f:Frame) --> (v:VersionedActors) --> (a:Actor) DETACH DELETE a,v,f'
                db.cypher_query(delete_query)
                # trunk-ignore(pylint/C0301)
                delete_query = f'match (n:DataCollection{{uuid:"{data_collection.uuid}"}}) -->(f:Frame) --> (a:Actor) DETACH DELETE a,f'
                db.cypher_query(delete_query)
            except DoesNotExist:
                db.rollback()
                logger.exception("Datacollection does not exist")
                return DataCollectionUpdate(
                    data_collection=None, success=False
                )
            except NeomodelException:  # Terminate for any other exception
                db.rollback()
                logger.exception(
                    "Error encountered in DataCollectionUpdate while querying data_collection"
                )
                return DataCollectionUpdate(
                    data_collection=None, success=False
                )
            try:
                if kwargs.get("frames") is not None:
                    for frame in kwargs["frames"]:
                        frame_val = Frame(**frame).save()
                        data_collection.frame_ref.connect(frame_val)
                        _add_actors(frame["actors"], frame_val)

                # Add label metadata
                if kwargs.get("label_metadata") is not None:
                    if not data_collection.label_metadata_ref:
                        kwargs["label_metadata"]["taxonomy_version"] = [
                            kwargs["label_metadata"]["taxonomy_version"]
                        ]
                        logger.info(
                            f'LabelMetadata not found {kwargs["label_metadata"]}'
                        )
                        label = LabelMetadata(
                            **(kwargs["label_metadata"])
                        ).save()
                        data_collection.label_metadata_ref.connect(label)
                    else:
                        label_metadata = (
                            data_collection.label_metadata_ref.all()[0]
                        )
                        if (
                            label_metadata.source
                            != kwargs["label_metadata"]["source"]
                            or label_metadata.project_name
                            != kwargs["label_metadata"]["project_name"]
                        ):
                            raise RuntimeError(
                                "Source and project not same for labels"
                            )
                        if not label_metadata.taxonomy_version:
                            label_metadata.taxonomy_version = [
                                kwargs["label_metadata"]["taxonomy_version"]
                            ]
                        else:
                            label_metadata.taxonomy_version.append(
                                kwargs["label_metadata"]["taxonomy_version"]
                            )
                            label_metadata.taxonomy_version = list(
                                set(label_metadata.taxonomy_version)
                            )
                        label_metadata.save()
                db.commit()
                success = True
            except NeomodelException:
                db.rollback()
                logger.exception(
                    "Exception encountered in DataCollectionUpdate while connecting "
                    "violations, frames and actors"
                )
                success = False
        except Exception:  # trunk-ignore(pylint/W0703)
            db.rollback()
            logger.exception("Error encountered in DataCollectionUpdate")
            success = False
        return DataCollectionUpdate(
            data_collection=data_collection, success=success
        )


class DataCollectionMutations(graphene.ObjectType):
    data_collection_create = DataCollectionCreate.Field()
    data_collection_update = DataCollectionUpdate.Field()


def _add_actors(actors_list, frame_val):
    if actors_list is not None:
        for actor in actors_list:
            actor_val = Actor(**actor).save()
            frame_val.actors_ref.connect(actor_val)
