#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

"""Central location for all metaverse queries.
"""
import copy
import json
import os
from typing import List, Optional
from uuid import uuid4

import git
from graphql.execution import ExecutionResult
from loguru import logger

from core.metaverse.metaverse import Metaverse
from core.structs.dataset import DataCollectionLogset, Dataset, DatasetFormat
from core.structs.model import ModelCategory
from core.structs.task import Task, TaskPurpose

# trunk-ignore-all(semgrep)


def get_result(
    query: str, metaverse_environment: Optional[str] = None
) -> ExecutionResult:
    """Result of the metaverse query

    Args:
        query (str): Query to run
        metaverse_environment (Optional[str]): metaverse environment to execute

    Returns:
        ExecutionResult: result of the query
    """
    metaverse = Metaverse(environment=metaverse_environment)
    return metaverse.schema.execute(query)


class DataCollectionLogsetQueryException(RuntimeError):
    """
    Generated when the data collection logset query fails
    """


def generate_data_collection_logset_from_query(
    query: str,
    query_name: str,
    query_vars: dict,
    logset_name: Optional[str] = None,
) -> DataCollectionLogset:
    """
    Generates a data collection logset from a metaverse query

    Args:
        query (str): the query to submit to metaverse
        query_name (str): the query name
        query_vars (str): the extra query variables
        logset_name (Optional[str]): the task uuid used in versioning the logset
                         if this is not available the default logset name is:
                             (unlinked)-<hash_of_query>

    Raises:
        RuntimeError: if the query failed. An error is returned

    Returns:
        DataCollectionLogset: the datacollection logset generated from the datacollection query
    """
    if logset_name is None:
        logger.warning(
            "Logset name was none, NOTE: this is only to be used for offline testing purposes"
        )

    metaverse = Metaverse()
    result = metaverse.schema.execute(query, variables=query_vars)
    if not result.data:
        raise DataCollectionLogsetQueryException(
            f"Data collection logset query: {query} failed with errors: {result}"
        )
    data_collections = [
        {"data_collection_uuid": uuid["uuid"]}
        for uuid in result.data[query_name]
    ]
    result = metaverse.schema.execute(
        """mutation create_data_collection_logset(
                $name: String,
                $data_collections: [DataCollectionReferenceArgument]
            ) {
            data_collection_logset_create(name: $name, data_collections:  $data_collections )
            { data_collection_logset { uuid, version }, success}
        }
        """,
        variables={
            "data_collections": data_collections,
            "name": f"(unlinked)-{str(uuid4())}"
            if logset_name is None
            else logset_name,
        },
    )
    if (
        not result.data
        or not result.data["data_collection_logset_create"]
        or not result.data["data_collection_logset_create"]["success"]
    ):
        raise RuntimeError(
            f"Could not create logset with error {result.errors}"
        )
    return DataCollectionLogset(
        **result.data["data_collection_logset_create"][
            "data_collection_logset"
        ]
    )


def is_data_collection_in_metaverse(
    data_collection_uuid: str, metaverse_environment: Optional[str] = None
):
    """Checks if data_collection is in metaverse

    Args:
        data_collection_uuid (str): name of data_collection to query metaverse for
        metaverse_environment (Optional[str]): metaverse environment to check for data collection

    Raises:
        RuntimeError: If the query fails.

    Returns:
        bool: whether the data_collection is in the metaverse
    """
    metaverse = Metaverse(environment=metaverse_environment)

    query = """query get_data_collection_from_voxel_uuid($voxel_uuid: String!) {
            data_collection_from_voxel_uuid(voxel_uuid:$voxel_uuid) {
                uuid, is_test
            }
        }
        """
    qvars = {"voxel_uuid": data_collection_uuid}
    result = metaverse.schema.execute(query, variables=qvars)
    data = result.data["data_collection_from_voxel_uuid"]

    # Catch and raise if there is a query errors
    if result.errors is not None:
        raise RuntimeError(f"Query error: {result.errors}")

    # Return true if the data_collection is in the Metaverse
    if data:
        return True

    # otherwise return false
    return False


def create_task(
    purpose: TaskPurpose,
    camera_uuids: List,
    metaverse_environment: Optional[str] = None,
) -> dict:
    """Creates a new task for camera and purpose

    Args:
        purpose (TaskPurpose): purpose of the task
        camera_uuids (List): camera uuid the task belongs to
        metaverse_environment (Optional[str]): metaverse environment
    Raises:
        RuntimeError: when the task creation failed

    Returns:
        dict: Result of creating node in metaverse.
    """
    logger.info("Creating task in model registry")
    query = f'mutation {{ task_create(purpose:"{purpose.name}", \
        camera_uuids: {json.dumps(camera_uuids)}){{task {{ uuid, purpose }} }}}}'
    result = get_result(query, metaverse_environment=metaverse_environment)
    if not result.data:
        raise RuntimeError(f"Task creation failed! Result:\n {result}")
    logger.info(f'Task result is {result.data["task_create"]["task"]}')
    return result.data["task_create"]["task"]


def query_task(
    purpose: TaskPurpose,
    camera_uuids: List,
    metaverse_environment: Optional[str] = None,
) -> dict:
    """Queries for a task in metaverse

    Args:
        purpose (TaskPurpose): purpose of the task
        camera_uuids (List): camera uuid the task belongs to
        metaverse_environment (Optional[str]): metaverse environment to query task
    Returns:
        dict: Result of querying task node in metaverse.
    """
    logger.info("Fetching task in model registry")
    query = f'{{task_from_cameras(purpose:"{purpose.name}", camera_uuids: \
        {json.dumps(camera_uuids)}) {{  uuid, purpose, service_ref {{ uuid, category}}}} }}'
    task_result = get_result(
        query, metaverse_environment=metaverse_environment
    )
    return task_result.data["task_from_cameras"]


def create_service(
    task_uuid: str,
    category: ModelCategory,
    metaverse_environment: Optional[str] = None,
) -> dict:
    """Creates a service in metaverse given a task uuid and category.

    Args:
        task_uuid (str): Uuid of the task
        category (ModelCategory): Category of the service/model
        metaverse_environment (Optional[str]): metaverse environment

    Raises:
        RuntimeError: If the service was not created.

    Returns:
        dict: Service details
    """
    query = """mutation service_create(
            $category: String,
            $task_uuid: String,
    )
    {
        service_create(
            category:$category,
            task_uuid:$task_uuid
        ) { service { uuid } }
    }"""

    qvars = {
        "task_uuid": task_uuid,
        "category": category.name,
    }
    result = Metaverse(environment=metaverse_environment).schema.execute(
        query, variables=qvars
    )
    if not result.data or not result.data["service_create"]:
        raise RuntimeError(
            f"Failed to create service with error {result.errors}"
        )
    return result.data["service_create"]["service"]


def get_or_create_task_and_service(
    purpose: TaskPurpose,
    category: ModelCategory,
    camera_uuids: List,
    metaverse_environment: Optional[str] = None,
) -> Task:
    """Gets task corresponding to metadata.

    Args:
        purpose (TaskPurpose): purpose of the task
        category (ModelCategory): category of the service
        camera_uuids (List): camera uuid the task belongs to
        metaverse_environment (Optional[str]): metaverse environment to run query
    Raises:
        RuntimeError: If the task was neither created not present in metaverse.

    Returns:
        Task: Task for the purpose and camera uuids
    """
    if len(camera_uuids) == 0:
        if purpose != TaskPurpose.OBJECT_DETECTION_2D:
            raise RuntimeError(
                "Only OBJECT_DETECTION_2D tasks support empty camera uuids"
            )
    task_result = query_task(
        purpose, camera_uuids, metaverse_environment=metaverse_environment
    )
    if not task_result:
        task_result = create_task(
            purpose, camera_uuids, metaverse_environment=metaverse_environment
        )
    if not task_result.get("uuid"):
        raise RuntimeError("Task not present, neither created")

    # Getting or creating a service.
    if not task_result.get("service_ref") or category.name not in [
        service["category"] for service in task_result["service_ref"]
    ]:
        service = create_service(
            task_result["uuid"],
            category,
            metaverse_environment=metaverse_environment,
        )
        service_id = service["uuid"]
    else:
        service_ids = [
            service["uuid"]
            for service in task_result["service_ref"]
            if service["category"] == category.name
        ]
        if len(service_ids) != 1:
            raise RuntimeError(
                f"Number of service with same category for a task"
                f" should be one found {len(service_ids)}"
            )
        service_id = service_ids[0]

    camera_query = (
        f'{{cameras_for_task(task_uuid:"{task_result["uuid"]}") {{ uuid }} }}'
    )
    camera_result = Metaverse(
        environment=metaverse_environment
    ).schema.execute(camera_query)
    if (
        not camera_result.data
        or camera_result.data["cameras_for_task"] is None
    ):
        raise RuntimeError(
            f"Unable to get camera for task {camera_result.data}"
        )
    camera_uuids = list(
        {camera["uuid"] for camera in camera_result.data["cameras_for_task"]}
    )
    return Task(
        camera_uuids=camera_uuids,
        service_id=service_id,
        model_category=category.name,
        **task_result,
    )


def get_dataset(uuid: str) -> Dataset:
    """
    Grabs a dataset from metaverse given a uuid

    Args:
        uuid (str): the uuid of the dataset to be grabbed

    Returns:
        Dataset: a dataset object constructed dynamically from the result of the
                 graphql query
    Raises:
        RuntimeError: when the metaverse query failed. Graphql error messages are provided
    """
    query = """ query get_dataset($uuid: String){
    dataset(uuid: $uuid) {
        uuid
        created_timestamp
        config
        path
        format
        data_collection_logset_ref
        {
            uuid
            name
            version
            data_collection
            {
                data_collection_uuid
                name
                path
            }
        }
        git_version
        metadata
        version
    }}
    """

    metaverse = Metaverse()
    qvars = {"uuid": uuid}
    result = metaverse.schema.execute(query, variables=qvars)
    if result.data is None:
        logger.error("Query failed, please check uuid")
        logger.error(result.errors)
        raise RuntimeError(f"Query failed for uuid: {uuid}")
    return Dataset(**result.data["dataset"][0])


def register_model(
    metadata: dict,
    model_config: dict,
    run_links: List[str],
    path: str,
    dataset_uuid: str,
    metrics: dict,
    task: Task,
) -> dict:
    """
    Create Metaverse model object

    Arguments:
        metadata (dict): training metadata
        model_config (dict): configuration of model
        run_links (List[str]): list of URLs to model information (e.g. clearml information)
        path (str): URL to model file
        dataset_uuid(str): Uuid of dataset to link to
        metrics(dict): model metrics using test data
        task (Task): the task required to register the model

    Returns:
        dict: details of model registered

    Raises:
        RuntimeError: If the model was not created or was not added to a service
    """
    model_config = copy.copy(model_config)
    task = copy.deepcopy(task)

    model_query = """mutation createModel(
            $name: String,
            $metadata: JSONString,
            $path: String,
            $config: JSONString,
            $run_links: [String],
            $dataset_uuid: String,
            $metrics: JSONString
    ) {
        model_create(
            name: $name,
            metadata: $metadata,
            path: $path,
            run_links: $run_links,
            config: $config,
            dataset_uuid: $dataset_uuid,
            metrics: $metrics
        ) { model { uuid, path }, success, error }
        }"""

    model_qvars = {
        "name": metadata["name"],
        "path": path,
        "metadata": json.dumps(metadata),
        "config": json.dumps(model_config),
        "run_links": run_links,
        "dataset_uuid": dataset_uuid,
        "metrics": json.dumps(metrics),
    }
    metaverse = Metaverse()
    model_result = metaverse.schema.execute(model_query, variables=model_qvars)
    if not model_result.data or not model_result.data["model_create"]:
        raise RuntimeError(
            f"Could not register the model with error {model_result.errors}"
        )

    add_model_to_service_query = """mutation addModelToService(
        $service_uuid: String,
        $model_uuids: [String],
    ) {
        service_add_models(
            service_uuid: $service_uuid,
            model_uuids: $model_uuids
        ) {
            success, error
        }
    }
    """
    add_model_to_service_qvars = {
        "service_uuid": task.service_id,
        "model_uuids": [model_result.data["model_create"]["model"]["uuid"]],
    }
    service_result = metaverse.schema.execute(
        add_model_to_service_query, variables=add_model_to_service_qvars
    )
    if (
        not service_result.data
        or not service_result.data["service_add_models"]
        or not service_result.data["service_add_models"]["success"]
    ):
        raise RuntimeError(
            f"Could not add model to service with error {service_result.errors}"
        )
    return model_result.data["model_create"]["model"]


def register_dataset(
    config: dict,
    cloud_path: str,
    logset: DataCollectionLogset,
    dataset_format: DatasetFormat,
) -> Dataset:
    """
    Registers the dataset in metaverse

    Args:
        config (dict): the config used to generate the dataset
        cloud_path (str): the cloud path of the dataset directory
        logset (DataCollectionLogset): the datacollection logset
                                            used to generate the dataset
        dataset_format (DatasetFormat): the dataset format used to generate the dataset
                                        this is used to assist generating dataloaders

    Raises:
        RuntimeError: if the dataset could not be registered. Returns the error

    Returns:
        Dataset: the registered dataset
    """

    metaverse = Metaverse()
    s3_path = str(uuid4())

    git_version = (
        git.Repo(
            os.environ["BUILD_WORKSPACE_DIRECTORY"],
            search_parent_directories=True,
        ).head.object.hexsha
        if "BUILD_WORKSPACE_DIRECTORY" in os.environ
        else None
    )
    s3_path = cloud_path
    # now we sync everything in the dataset
    json_config = json.dumps(config)
    dataset_query = """mutation createDataset(
            $config: JSONString,
            $metadata: JSONString,
            $path: String,
            $logset_uuid: String,
            $git_version: String,
            $format: String,
    ) {
        dataset_create(
            config: $config,
            path: $path,
            format: $format,
            metadata: $metadata,
            logset_uuid: $logset_uuid,
            git_version: $git_version,
        ) { dataset { uuid }, success, error }
        }"""
    dataset_result = metaverse.schema.execute(
        dataset_query,
        variables={
            "config": json_config,
            "path": s3_path,
            "git_version": git_version,
            "metadata": json.dumps({}),
            "logset_uuid": logset.uuid,
            "format": dataset_format.name,
        },
    )
    if (
        not dataset_result.data
        or not dataset_result.data["dataset_create"]
        or not dataset_result.data["dataset_create"]["success"]
    ):
        if dataset_result.data and dataset_result.data["dataset_create"]:
            logger.error(dataset_result["dataset_create"]["error"])
        raise RuntimeError(
            f"Could not add model to service with error {dataset_result.errors}"
        )

    dataset_uuid = dataset_result.data["dataset_create"]["dataset"]["uuid"]
    dataset = get_dataset(dataset_uuid)
    return dataset
