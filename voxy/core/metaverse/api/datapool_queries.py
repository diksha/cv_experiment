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

import json
import os
from typing import Optional

from lightly.api import ApiWorkflowClient
from loguru import logger

from core.metaverse.metaverse import Metaverse
from core.ml.data.curation.voxel_lightly_utils import (
    LIGHTLY_TOKEN_ARN,
    DatasetType,
    create_and_configure_dataset,
)
from core.structs.datapool import Datapool
from core.structs.task import Task
from core.utils.aws_utils import get_secret_from_aws_secret_manager


class DatapoolQueryException(Exception):
    """
    Raised when the datapool queries to create or find the datapool
    """


def update_datapool(
    uuid: str,
    ingested_data_collections: list,
    metaverse_environment: Optional[str] = None,
) -> bool:
    """Updates datapool with ingested datacollections

    Args:
        uuid (str): uuid of datapool
        ingested_data_collections (list): voxel uuid of ingested datacollection
        metaverse_environment (Optional[str]): metaverse environment to update datapool

    Raises:
        DatapoolQueryException: raised when the datapool creation query fails

    Returns:
        bool: datapool is updated or not
    """
    result = Metaverse(environment=metaverse_environment).schema.execute(
        """
        mutation updateDatapool(
            $uuid: String,
            $ingested_data_collections: [String],
        )
        {
            datapool_update(uuid: $uuid,ingested_data_collections: $ingested_data_collections)
            {
                datapool { name }
            }
        }
        """,
        variables={
            "uuid": uuid,
            "ingested_data_collections": ingested_data_collections,
        },
    )
    logger.info(f"Datapool update result {result.data}")
    if not result.data or not result.data.get("datapool_update"):
        raise DatapoolQueryException(
            f"Could not update datapool, failed with exeception: {result}"
        )
    return True


def get_or_create_datapool_from_task(
    task: Task,
    config: dict,
    metaverse_environment: Optional[str] = None,
) -> Datapool:
    """
    Simple helper to generate the datapool from a task

    TODO: make this not a stub and generate this from the
    registry

    Args:
        task (Task): the task required to generate the datapool
        config (dict): the configuration to run the datapool in lightly
        metaverse_environment (Optional[str]): metaverse environment to run query
    Raises:
        DatapoolQueryException: raised when the queries to query
                            the service or create the datapool fail.

    Returns:
        Datapool: the datapool pulled from metaverse
    """
    dataset_name = f"{task.purpose}_{task.model_category}_{task.uuid}"
    output_directory = os.path.join("datapool_output", dataset_name)
    input_directory = os.path.join("datapool_input", dataset_name)

    # let's see if the datapool exists yet
    metaverse = Metaverse(environment=metaverse_environment)
    service_uuid = task.service_id
    datapool_query = """query getDatapoolsFromService($service_uuid: String)
            {
                datapool_from_service(service_uuid: $service_uuid)
                {
                        uuid
                        name
                        url
                        metadata
                        lightly_config
                        version
                        input_directory
                        output_directory
                        dataset_type
                        lightly_uuid
                        ingested_data_collections
                }
            }
        """
    result = metaverse.schema.execute(
        datapool_query, variables={"service_uuid": service_uuid}
    )
    if (
        not result
        or not result.data
        or "datapool_from_service" not in result.data
    ):
        raise DatapoolQueryException(
            f"Failed to create a datapool with result: {result}"
        )
    logger.info(result)

    if not result.data or not result.data.get("datapool_from_service"):
        logger.warning(
            f"No datapool found. Creating one with dataset name: {dataset_name}"
        )
        datapool = create_datapool(
            dataset_name,
            config,
            input_directory,
            output_directory,
            "VIDEOS",
            service_uuid,
            metaverse_environment=metaverse_environment,
        )
        logger.info(
            # trunk-ignore(pylint/E1101)
            f"Created datapool with lightly uuid: {datapool.lightly_uuid}"
        )
    else:
        # we get the most recent versioned datapool
        logger.info(
            f"Found {len(result.data['datapool_from_service'])} datapools"
        )
        datapool_dict = max(
            result.data["datapool_from_service"],
            key=lambda datapool: datapool["version"],
        )
        datapool = Datapool(**datapool_dict)

        logger.info(
            # trunk-ignore(pylint/E1101)
            f"Chose datapool with lightly uuid: {datapool.lightly_uuid}"
            # trunk-ignore(pylint/E1101)
            f" with latest version: {datapool.version}"
        )
    logger.info(
        f"Found datapool with contents: \n{json.dumps(datapool.to_dict(), indent=4)}"
    )
    return datapool


def create_datapool(
    name: str,
    lightly_config: dict,
    input_directory: str,
    output_directory: str,
    dataset_type: str,
    service_uuid: str,
    metaverse_environment: Optional[str] = None,
) -> Datapool:
    """
    Creates a datapool given the input arguments. Registers it with the lightly
    api and stores the result in metaverse

    Args:
        name (str): the indended name of the dataset
        lightly_config (dict): the config to pass to the lightly worker
        input_directory (str): the input directory
        output_directory (str): the output directory
        dataset_type (str): the dataset type from lightly. Either IMAGES, or VIDEOS
        service_uuid (str): the service ID to find the datapool
        metaverse_environment (Optional[str]): metaverse environment
    Raises:
        DatapoolQueryException: raised when the datapool creation query fails

    Returns:
        Datapool: the datapool pulled from metaverse
    """
    lightly_client = ApiWorkflowClient(
        token=json.loads(
            get_secret_from_aws_secret_manager(LIGHTLY_TOKEN_ARN)
        )["1"]
    )
    lightly_uuid = create_and_configure_dataset(
        lightly_client,
        name,
        input_directory,
        output_directory,
        dataset_type=getattr(DatasetType, dataset_type),
        metadata={"Datapool": "created"},
        notify=False,
    )
    lightly_url = f"https://app.lightly.ai/dataset/{lightly_uuid}"
    metaverse = Metaverse(environment=metaverse_environment)
    result = metaverse.schema.execute(
        """
        mutation createDatapool(
            $service_uuid: String,
            $name: String,
            $url: String,
            $lightly_uuid: String,
            $config: JSONString,
            $metadata: JSONString,
            $dataset_type: String,
            $input_directory: String,
            $output_directory: String,
        )
        {
            datapool_create(service_uuid: $service_uuid,
                            name: $name,
                            url: $url,
                            metadata: $metadata,
                            dataset_type: $dataset_type,
                            input_directory: $input_directory,
                            output_directory: $output_directory,
                            lightly_uuid: $lightly_uuid,
                            lightly_config: $config)
                            {
                                datapool
                                    {
                                        uuid
                                        name
                                        url
                                        metadata
                                        lightly_config
                                        version
                                        input_directory
                                        output_directory
                                        dataset_type
                                        lightly_uuid
                                        ingested_data_collections
                                    }
                            }
        }
        """,
        variables={
            "service_uuid": service_uuid,
            "name": name,
            "url": lightly_url,
            "lightly_uuid": lightly_uuid,
            "config": json.dumps(lightly_config),
            "metadata": json.dumps({}),
            "dataset_type": dataset_type,
            "input_directory": input_directory,
            "output_directory": output_directory,
        },
    )
    logger.info(result)
    if not result.data or not result.data.get("datapool_create"):
        raise DatapoolQueryException(
            f"Could not create datapool, failed with exeception: {result}"
        )
    datapool_dict = result.data["datapool_create"]["datapool"]
    return Datapool(**datapool_dict)
