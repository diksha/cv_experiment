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
from loguru import logger
from neomodel import db

from core.metaverse.models.model import (
    DataCollectionLogset,
    Datapool,
    Dataset,
    DatasetCollection,
    Model,
    Service,
    Task,
)

# trunk-ignore-all(pylint/R0903)
# trunk-ignore-all(pylint/W9012)
# trunk-ignore-all(pylint/W9011)
# trunk-ignore-all(pylint/C0115)
# trunk-ignore-all(pylint/C0116)
# trunk-ignore-all(pylint/W0613)


class DataCollectionReferenceSchema(graphene.ObjectType):
    """Schema on what information from dataset collections is available to the user."""

    data_collection_uuid = graphene.String()
    name = graphene.String()
    path = graphene.String()


class DataCollectionLogsetSchema(graphene.ObjectType):
    """Schema on what information from dataset collections is available to the user."""

    uuid = graphene.String()
    name = graphene.String()
    version = graphene.Int()
    data_collection_uuid_hash = graphene.String()
    data_collection = graphene.List(DataCollectionReferenceSchema)


class DatasetSchema(graphene.ObjectType):
    uuid = graphene.String()
    created_timestamp = graphene.DateTime()
    config = graphene.JSONString()
    path = graphene.String()
    git_version = graphene.String()
    format = graphene.String()

    logset_type = graphene.String()
    # TODO: remove the video logset/image logset references
    data_collection_logset_ref = graphene.List(DataCollectionLogsetSchema)
    metadata = graphene.JSONString()
    version = graphene.String()


class ModelSchema(graphene.ObjectType):
    uuid = graphene.String()
    name = graphene.String()
    run_links = graphene.List(graphene.String)
    path = graphene.String()
    config = graphene.JSONString()
    dataset_ref = DatasetSchema
    metadata = graphene.JSONString()


class DatapoolSchema(graphene.ObjectType):
    uuid = graphene.String()
    created_timestamp = graphene.DateTime()
    name = graphene.String()
    url = graphene.String()
    lightly_uuid = graphene.String()
    metadata = graphene.JSONString()
    lightly_config = graphene.JSONString()
    version = graphene.Int()
    input_directory = graphene.String()
    output_directory = graphene.String()
    dataset_type = graphene.String()
    ingested_data_collections = graphene.List(graphene.String)


class ServiceSchema(graphene.ObjectType):
    uuid = graphene.String()
    created_timestamp = graphene.DateTime()
    category = graphene.String()
    metadata = graphene.JSONString()
    model_refs = graphene.List(ModelSchema)
    datapool_ref = graphene.List(DatapoolSchema)

    best_model_ref = ModelSchema
    latest_model_ref = ModelSchema
    version = graphene.Int()


class TaskSchema(graphene.ObjectType):
    uuid = graphene.String()
    purpose = graphene.String()
    metadata = graphene.String()
    service_ref = graphene.List(ServiceSchema)


class DatasetCollectionSchema(graphene.ObjectType):
    uuid = graphene.String()
    created_timestamp = graphene.DateTime()
    metadata = graphene.JSONString()
    dataset_refs = graphene.List(DatasetSchema)
    task_ref = TaskSchema


class ModelQueries(graphene.ObjectType):
    model = graphene.List(
        ModelSchema,
        name=graphene.String(),
    )

    def resolve_model(self, info, *args, **kwargs):
        return Model().nodes.filter(**kwargs)


class TaskQueries(graphene.ObjectType):
    task_from_cameras = graphene.Field(
        TaskSchema,
        purpose=graphene.String(),
        camera_uuids=graphene.List(graphene.String),
    )

    def resolve_task_from_cameras(self, _, *__, **kwargs):
        """Returns a task given camera_uuids and purpose. For some tasks camera_uuid
        is optional since it might belong to all camera_uuids.

        Args:
            kwargs:
                purpose: Purpose of the task
                camera_uuids: UUIDs of the camera
            __: unused


        Raises:
            RuntimeError: If there are more than one task matching the query

        Returns:
            Task: task corresponding the arguments
        """
        uuid = None

        def get_task_for_query(query: str) -> str:
            results, _ = db.cypher_query(query)
            if len(results) != 1 or len(results[0]) != 1:
                logger.error(
                    f"There should only be one task matching the arguments {results}"
                )
                raise RuntimeError(
                    "There should only be one task matching the arguments"
                    + str(results)
                )
            return results[0][0]

        for camera in kwargs.get("camera_uuids", []):
            query = f'match(n:Task)--(c:Camera) where n.purpose="{kwargs.get("purpose")}" \
                and c.uuid="{camera}" return distinct n.uuid'
            uuid_local = get_task_for_query(query)
            if not uuid:
                uuid = uuid_local
            elif uuid != uuid_local:
                raise RuntimeError(
                    "Tasks for camera uuids differ. Cannot get 1 task for all camera_uuids"
                )

        if not kwargs.get("camera_uuids", []):
            query = f'match(n:Task) where n.purpose="{kwargs.get("purpose")}" \
                 return distinct n.uuid'
            uuid = get_task_for_query(query)
        if not uuid:
            raise RuntimeError("Could not get task for the given arguments")
        task = Task().nodes.get(uuid=uuid)
        return task


class ServiceQueries(graphene.ObjectType):
    service = graphene.List(
        ServiceSchema,
        name=graphene.String(),
    )

    def resolve_service(self, info, *args, **kwargs):
        return Service().nodes.filter(**kwargs)


class DatapoolQueries(graphene.ObjectType):
    datapool = graphene.List(
        DatapoolSchema,
        name=graphene.String(),
    )

    datapool_from_service = graphene.List(
        DatapoolSchema,
        service_uuid=graphene.String(),
    )

    def resolve_datapool(self, info, *args, **kwargs):
        return Datapool().nodes.filter(**kwargs)

    def resolve_datapool_from_service(self, info, *args, **kwargs):
        # make a query in metaverse
        service_uuid = kwargs["service_uuid"]
        query = (
            f"MATCH (d: Datapool)--(s:Service{{uuid: '{service_uuid}'}})"
            " RETURN distinct d.uuid"
        )
        results, _ = db.cypher_query(query)
        uuids = [uuid[0] for uuid in results]
        return Datapool().nodes.filter(uuid__in=uuids)


class DatasetQueries(graphene.ObjectType):
    dataset = graphene.List(
        DatasetSchema,
        uuid=graphene.String(),
    )

    dataset_collection = graphene.List(
        DatasetCollectionSchema,
        uuid=graphene.String(),
    )

    def resolve_dataset(self, info, *args, **kwargs):
        return Dataset().nodes.filter(**kwargs)

    def resolve_dataset_collection(self, info, *args, **kwargs):
        return DatasetCollection().nodes.filter(**kwargs)


class LogsetQueries(graphene.ObjectType):
    data_collection_logset = graphene.Field(
        DataCollectionLogsetSchema,
        name=graphene.String(),
        version=graphene.Int(),
    )

    def resolve_data_collection_logset(
        self, info, *args, **kwargs
    ):  # pylint: disable=unused-argument
        return DataCollectionLogset().nodes.get(**kwargs)
