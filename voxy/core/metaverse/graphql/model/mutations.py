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
import hashlib
from typing import Any, List

import graphene
from loguru import logger
from neomodel import DoesNotExist, NeomodelException, db

from core.metaverse.graphql.model.schema import (
    DataCollectionLogsetSchema,
    DatapoolSchema,
    DatasetSchema,
    ModelSchema,
    ServiceSchema,
    TaskSchema,
)
from core.metaverse.models.camera import Camera
from core.metaverse.models.model import (
    DataCollectionLogset,
    DataCollectionReference,
    Datapool,
    Dataset,
    Model,
    Service,
    Task,
)
from core.structs.dataset import DatasetFormat
from core.structs.model import ModelCategory
from core.structs.task import TaskPurpose

# NOTE: Trunk doesn't like the way that the current graphql schemas are defined
#       because of missing docstrings. Since these are not ever to be instantiated
#       and only link to the graphql api. These will ignore them for now. In the future,
#       these should auto create documentation to better track the graphql apis.

# trunk-ignore-all(pylint/R0903)
# trunk-ignore-all(pylint/C0115)
# trunk-ignore-all(pylint/C0116)
# trunk-ignore-all(pylint/W9011)


class TaskCreate(graphene.Mutation):
    class Arguments:
        purpose = graphene.String(required=True)
        metadata = graphene.JSONString()
        camera_uuids = graphene.List(graphene.String, required=True)

    success = graphene.Boolean()
    task = graphene.Field(TaskSchema)
    error = graphene.String()

    def mutate(
        self,
        info: graphene.ResolveInfo,
        **kwargs: Any,
    ) -> "TaskCreate":
        """
        Creates a task given a set of keyword arguments. If the camera already has a
        task connected with the given purpose, then the task is not created and an error
        is raised

        Args:
            info (graphene.ResolveInfo): the resolver info for graphql
            kwargs (Any): keyword arguments for the task creation mutation

        Returns:
            TaskCreate: The task creation object
        """
        db.begin()
        try:
            purpose, metadata, camera_uuids = (
                kwargs["purpose"],
                kwargs.get("metadata"),
                kwargs["camera_uuids"],
            )

            error = None
            if purpose not in TaskPurpose.names():
                error = f"Purpose: {purpose} not found in available purposes: {TaskPurpose.names()}"
                raise NeomodelException(error)

            valid_camera_references = []
            # only create the task if all the cameras are correct
            for camera_uuid in camera_uuids:
                camera_ref = _check_or_create_camera(camera_uuid)
                # check the camera reference
                if not _check_if_camera_has_task_purpose(camera_ref, purpose):
                    valid_camera_references.append(camera_ref)
                else:
                    logger.warning(
                        f"Found {camera_uuid} that already has a task with the same purpose"
                    )
            if len(valid_camera_references) != len(camera_uuids):
                error = (
                    f"Found task with purpose {purpose} in list\
                     of camera uuid nodes: {camera_uuids}",
                )
                raise NeomodelException(error)

            # first validate that the cameras are defined correctly
            task = Task(purpose=purpose, metadata=metadata).save()
            for camera_ref in valid_camera_references:
                camera_ref.task_ref.connect(task)

        except NeomodelException as error:
            logger.info(f"error was encountered {error}")
            error = (
                error if error is not None else "Neomodel raised an exception"
            )
            db.rollback()
            return TaskCreate(task=None, success=False, error=error)
        db.commit()
        success = True
        return TaskCreate(task=task, success=success)


class DatasetCreate(graphene.Mutation):
    class Arguments:
        config = graphene.JSONString()
        path = graphene.String()
        format = graphene.String()
        git_version = graphene.String()
        metadata = graphene.JSONString()
        logset_uuid = graphene.String()

    success = graphene.Boolean()
    dataset = graphene.Field(DatasetSchema)
    error = graphene.String(required=False)

    def mutate(
        self,
        info: graphene.ResolveInfo,  # trunk-ignore(pylint/W0613)
        **kwargs: Any,
    ) -> "DatasetCreate":
        db.begin()
        error = None
        try:
            if kwargs.get("format") not in DatasetFormat.names():
                dataset_format = kwargs.get("format")
                error = (
                    f"Format: {dataset_format} not found in "
                    f"available dataset format types: {DatasetFormat.names()}"
                )
                raise NeomodelException(error)

            dataset = Dataset(
                config=kwargs.get("config"),
                metadata=kwargs.get("metadata"),
                path=kwargs.get("path"),
                git_version=kwargs.get("git_version"),
                format=kwargs.get("format"),
            ).save()

            logset_node = DataCollectionLogset().nodes.get(
                uuid=kwargs.get("logset_uuid")
            )

            if not logset_node:
                error = (
                    f"Could not find logset uuid: {kwargs.get('logset_uuid')}"
                )
                raise NeomodelException(error)

            dataset.data_collection_logset_ref.connect(logset_node)
        except NeomodelException:
            db.rollback()
            return DatasetCreate(
                dataset=None,
                success=False,
                error=error if error else "Neomodel encountered an exception",
            )
        db.commit()
        success = True
        return DatasetCreate(dataset=dataset, success=success)


class DatapoolUpdate(graphene.Mutation):
    class Arguments:
        ingested_data_collections = graphene.List(graphene.String)
        uuid = graphene.String()

    success = graphene.Boolean()
    datapool = graphene.Field(DatapoolSchema)

    def mutate(
        self,
        info: graphene.ResolveInfo,
        **kwargs: Any,
    ) -> "DatapoolUpdate":
        db.begin()
        try:
            datapool = Datapool().nodes.get(uuid=kwargs.get("uuid"))
            if not datapool:
                error = f"Could not find datapool : {kwargs.get('uuid')}"
                raise NeomodelException(error)
            if not datapool.ingested_data_collections:
                datapool.ingested_data_collections = []
            datapool.ingested_data_collections.extend(
                kwargs.get("ingested_data_collections")
            )
            datapool.save()
        except NeomodelException as exception:
            db.rollback()
            raise exception
        db.commit()
        success = True
        return DatapoolUpdate(datapool=datapool, success=success)


class DatapoolCreate(graphene.Mutation):
    class Arguments:
        service_uuid = graphene.String()
        name = graphene.String()
        url = graphene.String()
        lightly_uuid = graphene.String()
        metadata = graphene.JSONString()
        lightly_config = graphene.JSONString()
        input_directory = graphene.String()
        output_directory = graphene.String()
        dataset_type = graphene.String()

    success = graphene.Boolean()
    datapool = graphene.Field(DatapoolSchema)
    error = graphene.String(required=False)

    def mutate(
        self,
        info: graphene.ResolveInfo,
        **kwargs: Any,
    ) -> "DatapoolCreate":
        db.begin()
        error = None
        try:
            service_node = Service().nodes.get(uuid=kwargs.get("service_uuid"))

            if not service_node:
                error = f"Could not find service_node : {kwargs.get('service_uuid')}"
                raise NeomodelException(error)
            kwargs.pop("service_uuid")
            datapool_version = 0
            for datapool_ref in service_node.datapool_ref:
                datapool_version = max(datapool_ref.version, datapool_version)
            datapool_version += 1
            # first check to see if the logset exists
            datapool = Datapool(
                version=datapool_version,
                **kwargs,
            ).save()

            service_node.datapool_ref.connect(datapool)
        except NeomodelException as exception:
            db.rollback()
            raise exception
        db.commit()
        success = True
        return DatapoolCreate(datapool=datapool, success=success)


class ModelCreate(graphene.Mutation):
    class Arguments:
        name = graphene.String()
        run_links = graphene.List(graphene.String)
        path = graphene.String()

        dataset_uuid = graphene.String()
        config = graphene.JSONString()
        metadata = graphene.JSONString()
        metrics = graphene.JSONString()

    model = graphene.Field(ModelSchema)
    success = graphene.Boolean()
    error = graphene.String(required=False)

    def mutate(
        self,
        info: graphene.ResolveInfo,  # trunk-ignore(pylint/W0613)
        **kwargs: Any,
    ) -> "ModelCreate":
        db.begin()
        error = None
        try:
            model = Model(
                name=kwargs.get("name"),
                metadata=kwargs.get("metadata"),
                path=kwargs.get("path"),
                run_links=kwargs.get("run_links"),
                config=kwargs.get("config"),
                metrics=kwargs.get("metrics"),
            ).save()

            if kwargs.get("dataset_uuid") is not None:
                dataset_node = Dataset().nodes.get(uuid=kwargs["dataset_uuid"])

                if not dataset_node:
                    error = f"Could not find dataset uuid: {kwargs['dataset_uuid']}"
                    raise NeomodelException(error)
                model.dataset_ref.connect(dataset_node)
        except NeomodelException:
            db.rollback()
            return ModelCreate(
                model=None,
                success=False,
                error=error if error else "Neomodel encountered an error",
            )
        db.commit()
        success = True
        return ModelCreate(model=model, success=success)


class ServiceCreate(graphene.Mutation):
    class Arguments:
        category = graphene.String()
        metadata = graphene.JSONString()
        # model refs are added incrementally
        task_uuid = graphene.String()

    service = graphene.Field(ServiceSchema)
    success = graphene.Boolean()
    error = graphene.String(required=False)

    def mutate(
        self,
        info: graphene.ResolveInfo,  # trunk-ignore(pylint/W0613)
        **kwargs: Any,
    ) -> "ServiceCreate":
        db.begin()
        error = None
        try:
            # first check to see if the category is recorded:
            if kwargs.get("category") not in ModelCategory.names():
                error = (
                    f"Category {kwargs.get('category')} not found in "
                    f"ModelCategory: {ModelCategory.names()}"
                )
                raise NeomodelException(error)

            task_node = Task().nodes.get(uuid=kwargs.get("task_uuid"))

            if not task_node:
                error = f"Could not find task uuid: {kwargs.get('task_uuid')}"
                raise NeomodelException(error)

            # find the other service collection for this task
            service_refs = task_node.service_ref
            service_version = 0
            for service in service_refs:
                if service.category == kwargs.get("category"):
                    error = (
                        f"Service with category {kwargs.get('category')}"
                        " already present in task"
                    )
                    raise NeomodelException(error)
                service_version = max(service.version, service_version)
            service_version += 1

            service = Service(
                category=kwargs.get("category"),
                metadata=kwargs.get("metadata"),
                version=service_version,
            ).save()
            task_node.service_ref.connect(service)

        except NeomodelException:
            db.rollback()
            return ServiceCreate(
                model=None,
                success=False,
                error=error if error else "Neomodel encountered an error",
            )
        db.commit()
        success = True
        return ServiceCreate(service=service, success=success)


class ServiceUpdateAddModels(graphene.Mutation):
    class Arguments:
        service_uuid = graphene.String()
        model_uuids = graphene.List(graphene.String)

    success = graphene.Boolean()
    error = graphene.String(required=False)

    def mutate(
        self,
        info: graphene.ResolveInfo,  # trunk-ignore(pylint/W0613)
        **kwargs: Any,
    ) -> "ServiceUpdateAddModels":
        db.begin()
        error = None
        try:
            model_nodes = Model().nodes.filter(
                uuid__in=kwargs.get("model_uuids")
            )

            if not model_nodes or len(model_nodes) != len(
                kwargs.get("model_uuids", [])
            ):
                error = (
                    f"Could not find all models with uuids: {kwargs.get('model_uuids')}"
                    f"Found {model_nodes}"
                )
                raise NeomodelException(error)
            service_node = Service().nodes.get(uuid=kwargs.get("service_uuid"))
            if not service_node:
                error = f"Could not find any service nodes with uuid: {kwargs.get('service_uuid')}"
                raise NeomodelException(error)

            for model in model_nodes:
                service_node.model_refs.connect(model)
        except NeomodelException:
            db.rollback()
            return ServiceUpdateAddModels(
                success=False,
                error=error if error else "Neomodel encountered an error",
            )
        db.commit()
        return ServiceUpdateAddModels(success=True)


def _check_if_camera_has_task_purpose(camera: Camera, purpose: str) -> bool:
    """
    Checks to see if the camera already has the specific task purpose. Returns
    boolean

    Args:
        camera (Camera): the camera reference
        purpose (str): the purpose string to check

    Returns:
        bool: whether the camera has the task purpose
    """
    for task in camera.task_ref:
        if task.purpose == purpose:
            return True
    return False


def _check_or_create_camera(camera_uuid: str) -> Camera:
    """
    Checks to see if a camera exists, and if it does not, then it is created

    Args:
        camera_uuid (str): the camera uuid (organization/location/zone/channel)

    Returns:
        Camera: the camera node reference
    """
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
    return camera_ref


class DataCollectionReferenceArgument(graphene.InputObjectType):
    """Structure of argument passed to DataCollection"""

    data_collection_uuid = graphene.String()
    name = graphene.String()
    path = graphene.String()


# TODO(twroge): remove old get_video_logset_hash()
def get_data_collection_logset_hash(
    data_collections: List[DataCollectionReferenceArgument],
) -> str:
    """Get hash of data_collection UUIDs in logset

    Args:
        data_collections (List[DataCollectionReferenceArgument]): List of
                              data_collections to generate hash for

    Returns:
        str: SHA-256 hash of sorted, comma-separated data_collection UUIDs
    """
    logset_hash_input = ",".join(
        sorted(
            data_collection.data_collection_uuid
            for data_collection in data_collections
        )
    )
    return hashlib.sha256(logset_hash_input.encode("utf-8")).hexdigest()


class DataCollectionLogsetCreate(graphene.Mutation):
    """Allows to create a data_collection logset given a name and list of images"""

    class Arguments:
        name = graphene.String()
        data_collections = graphene.List(DataCollectionReferenceArgument)
        metadata = graphene.JSONString()

    success = graphene.Boolean()
    data_collection_logset = graphene.Field(DataCollectionLogsetSchema)

    def mutate(self, info, **kwargs) -> "DataCollectionLogsetCreate":
        db.begin()
        try:
            data_collection_logsets = DataCollectionLogset.nodes.filter(
                name=kwargs.get("name")
            )
            # Find the version of same named logset
            version, data_collection_logset = max(
                (
                    (data_collection_logset.version, data_collection_logset)
                    for data_collection_logset in data_collection_logsets
                ),
                default=(0, None),
                key=lambda x: x[0],
            )

            logset_hash = get_data_collection_logset_hash(
                kwargs.get("data_collections")
            )

            # Don't create duplicate of latest version
            if (
                data_collection_logset is None
                or logset_hash
                != data_collection_logset.data_collection_uuid_hash
            ):
                kwargs["version"] = version + 1
                kwargs["data_collection_uuid_hash"] = logset_hash
                data_collection_logset = DataCollectionLogset(**kwargs).save()
                for data_collection in kwargs.get("data_collections", []):
                    data_collection_ref = DataCollectionReference(
                        **data_collection
                    ).save()
                    data_collection_logset.data_collection.connect(
                        data_collection_ref
                    )
        except Exception as exception:
            logger.exception(
                "error was encountered in DataCollectionLogsetCreate"
            )

            db.rollback()
            raise exception
        db.commit()
        return DataCollectionLogsetCreate(
            data_collection_logset=data_collection_logset, success=True
        )


class ModelMutations(graphene.ObjectType):
    task_create = TaskCreate.Field()
    dataset_create = DatasetCreate.Field()
    model_create = ModelCreate.Field()
    service_create = ServiceCreate.Field()
    service_add_models = ServiceUpdateAddModels.Field()
    data_collection_logset_create = DataCollectionLogsetCreate.Field()
    datapool_create = DatapoolCreate.Field()
    datapool_update = DatapoolUpdate.Field()
