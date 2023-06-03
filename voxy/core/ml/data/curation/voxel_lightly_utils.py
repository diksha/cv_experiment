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
import os
import typing
from dataclasses import dataclass, field
from tempfile import NamedTemporaryFile
from typing import List

from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client.models.dataset_type import (
    DatasetType,
)
from lightly.openapi_generated.swagger_client.models.datasource_purpose import (
    DatasourcePurpose,
)
from loguru import logger

from core.utils.aws_utils import (
    get_secret_from_aws_secret_manager,
    upload_file,
)
from core.utils.logging.slack.get_slack_webhooks import (
    get_perception_verbose_sync_webhook,
)
from core.utils.logging.slack.synchronous_webhook_wrapper import (
    SynchronousWebhookWrapper,
)

LIGHTLY_TOKEN_ARN = (
    # trunk-ignore(bandit/B105)
    "arn:aws:secretsmanager:us-west-2:203670452561:"
    "secret:LIGHTLY_TOKEN-y7ninI"
)
LIGHTLY_DELEGATED_ACCESS_ROLE_ARN_TOKEN_ARN = (
    # trunk-ignore(bandit/B105)
    "arn:aws:secretsmanager:us-west-2:203670452561:"
    "secret:LIGHTLY_DELEGATED_ACCESS_ROLE_ARN-DCdCUt"
)
LIGHTLY_DELEGATED_ACCESS_EXTERNAL_ID_TOKEN_ARN = (
    # trunk-ignore(bandit/B105)
    "arn:aws:secretsmanager:us-west-2:203670452561:"
    "secret:LIGHTLY_DELEGATED_ACCESS_EXTERNAL_ID-tCrQIr"
)


VOXEL_LIGHTLY_EMAILS = [
    "harishma@voxelai.com",
    "vai@voxelai.com",
    "gabriel@voxelai.com",
    "diksha@voxelai.com",
    "nasha@voxelai.com",
    "jmalicki@voxelai.com",
    "tim@voxelai.com",
    "lightly-users@voxelai.com",
]


@dataclass
class LightlyVideoFrameSequence:
    """A summary of the frames that Lightly identified for labeling from a video

    Has fields exactly matching those in a sequence_information.json file from Lightly.

    See here for relevant Lightly docs:
    https://github.com/lightly-ai/lightly/blob/11c5c2b0f22a688c11258f98c32a2bcefc6c755b/docs/source/docker/advanced/sequence_selection.rst

    Attributes
    ----------
    video_name:
        The name of the video that was checked for labeling
    frame_names:
        A list of the filenames for the images pulled from the identified frames
    frame_indices:
        A list of the indices of the identified frames
    frame_timestamp_pts:
        The timestamps of the frames identified for labelling
    """

    video_name: str
    frame_names: List[str]
    frame_indices: List[int]
    frame_timestamps_pts: List[int]
    frame_timestamps_sec: List[float] = field(default_factory=list)


def notify_slack_channel_with_dataset(
    dataset_id: str, project: str, metadata: dict
):
    """
    Notifies the perception verbose channel with the information about the dataset

    Args:
        dataset_id (str): the lightly dataset id
        project (str): the project (i.e. "doors/cropped" or "detector")
        metadata (dict): metadata for slack notification
    """
    formatted_project = " ".join(
        [item.capitalize() for item in project.split("_")]
    )
    slack_webhook_notifier = SynchronousWebhookWrapper(
        get_perception_verbose_sync_webhook()
    )
    notification_block = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Lightly Dataset Generated, Project: "
                f"{formatted_project}",
            },
        }
    ]
    dataset_link = f"https://app.lightly.ai/dataset/{dataset_id}"
    embeddings_link = f"https://app.lightly.ai/dataset/{dataset_id}/embedding"
    logger.info(f"Dataset Link: {dataset_link}")
    logger.info(f"Embeddings Link: {embeddings_link}")

    link_fields = []
    link_fields.append(
        {
            "type": "mrkdwn",
            "text": f"Dataset Link: <{dataset_link}|{dataset_link}>",
        }
    )
    link_fields.append(
        {
            "type": "mrkdwn",
            "text": f"Embeddings Link: <{embeddings_link}|{embeddings_link}>",
        }
    )
    link_fields.append(
        {
            "type": "mrkdwn",
            "text": f"Metadata: {metadata}",
        }
    )
    notification_block.append(
        {
            "type": "section",
            "fields": link_fields,
        }
    )
    slack_webhook_notifier.post_message_block(notification_block)


class LightlyDatasetAccessException(RuntimeError):
    """
    Raised when the lightly buckets cannot get accessed:
      for `read`, `write`, or `list` access to the buckets
    """


def create_and_configure_dataset(
    client: ApiWorkflowClient,
    dataset_name: str,
    input_dir: str,
    output_dir: str,
    dataset_type: DatasetType = DatasetType.VIDEOS,
    aws_region: str = "us-west-2",
    metadata: dict = None,
    notify: bool = True,
) -> str:
    """Create and configure a Lightly video dataset for a given camera_uuid

    Args:
        client (ApiWorkflowClient): lightly client
        dataset_name (str): name of the dataset
        input_dir (str): directory to get data from
        output_dir (str): directory to store output to
        dataset_type (DatasetType, optional): Type of dataset. Defaults to DatasetType.VIDEOS.
        aws_region (str, optional): Defaults to "us-west-2".
        metadata (dict, optional): Metadata for dataset. Defaults to {}.
        notify (bool, optional): Whether or not to notify the dataset via slack. Defaults to True.

    Raises:
        RuntimeError:
            If the dataset access could not be verified with
            lightly for the input and output directories


    Returns:
        : returns the dataset_id of the newly created dataset
    """
    # TODO verify camera UUID is in correct format

    # Create Lightly Client
    input_bucket = "voxel-lightly-input"
    output_bucket = "voxel-lightly-output"

    # Create dataset name
    logger.info(f"Creating dataset: {repr(dataset_name)}")

    # Create a new dataset on the Lightly Platform.
    client.create_new_dataset_with_unique_name(
        dataset_name,
        dataset_type,
    )

    input_path = f"s3://{input_bucket}/{input_dir}"
    output_path = f"s3://{output_bucket}/{output_dir}"

    logger.info(
        f"Configuring input and output paths: {input_path} and {output_path}"
    )
    delegated_access_role_arn = get_secret_from_aws_secret_manager(
        LIGHTLY_DELEGATED_ACCESS_ROLE_ARN_TOKEN_ARN
    )
    delegated_access_external_id = get_secret_from_aws_secret_manager(
        LIGHTLY_DELEGATED_ACCESS_EXTERNAL_ID_TOKEN_ARN
    )

    client.set_s3_delegated_access_config(
        resource_path=input_path,
        region=aws_region,
        role_arn=delegated_access_role_arn,
        external_id=delegated_access_external_id,
        purpose=DatasourcePurpose.INPUT,
    )
    # Output bucket
    client.set_s3_delegated_access_config(
        resource_path=output_path,
        region=aws_region,
        role_arn=delegated_access_role_arn,
        external_id=delegated_access_external_id,
        purpose=DatasourcePurpose.LIGHTLY,
    )
    client.share_dataset_only_with(
        dataset_id=client.dataset_id, user_emails=VOXEL_LIGHTLY_EMAILS
    )

    # Verify
    verify_dataset_access = client._datasources_api.verify_datasource_by_dataset_id(  # trunk-ignore(pylint/W0212,pylint/C0301)
        client.dataset_id
    )
    logger.info(f"Verify dataset: {verify_dataset_access}")

    def has_access(dataset_access_result: dict) -> bool:
        """
        Checks to see if the lightly worker has access
        to the input S3 buckets, read/write/list access

        Args:
            dataset_access_result (dict): the result of the
                `verify_datasource_by_id` call in the lightly api

        Returns:
            bool: whether lightly has full access to the bucket
        """
        return (
            dataset_access_result["can_list"]
            and dataset_access_result["can_write"]
            and dataset_access_result["can_read"]
        )

    if not has_access(verify_dataset_access.to_dict()):
        logger.warning(
            f"Directory: {output_path} could not be found, writing to output path"
        )
        # likely this directory just doesn't exist, so let's try writing a file
        with NamedTemporaryFile() as tmp:
            with open(tmp.name, "w", encoding="utf-8") as output_file:
                output_file.write("test")
            test_output_file = upload_file(
                "voxel-lightly-output",
                tmp.name,
                os.path.join(output_dir, "test.txt"),
                extra_args={"ContentType": "text/plain"},
            )
        logger.warning(f"Wrote: {test_output_file}")

        logger.warning(
            f"Directory: {input_path} could not be found, writing to input path"
        )
        with NamedTemporaryFile() as tmp:
            with open(tmp.name, "w", encoding="utf-8") as output_file:
                output_file.write("test")
            test_output_file = upload_file(
                "voxel-lightly-input",
                tmp.name,
                os.path.join(input_dir, "test.txt"),
                extra_args={"ContentType": "text/plain"},
            )
        logger.warning(f"Wrote: {test_output_file}")
    verify_dataset_access = client._datasources_api.verify_datasource_by_dataset_id(  # trunk-ignore(pylint/W0212,pylint/C0301)
        client.dataset_id
    )
    if not has_access(verify_dataset_access.to_dict()):
        logger.warning(
            f"Dataset access could not be verified : {verify_dataset_access}."
            " Please check the input directory and output directory"
        )

    if notify:
        notify_slack_channel_with_dataset(
            dataset_id=client.dataset_id,
            project=dataset_name,
            metadata=metadata,
        )

    logger.info(f"Dataset id is {client.dataset_id}")

    return client.dataset_id


# trunk-ignore(pylint/R0913)
def get_or_create_existing_dataset_by_name(
    client: ApiWorkflowClient,
    dataset_id: typing.Optional[str],
    dataset_name: str,
    input_dir: str,
    output_dir: str,
    dataset_type: DatasetType = DatasetType.VIDEOS,
    aws_region: str = "us-west-2",
    metadata: dict = None,
    notify: bool = True,
) -> str:
    """
    Gets or creates an existing dataset by name. If the dataset
    exists, then the input arguments are checked against the one
    found in lightly. If the dataset params do not match, then an
    exception is raised

    Args:
        client (ApiWorkflowClient): lightly client
        dataset_id (typing.Optional[str]): the dataset id to lookup.
                     If None, then a new dataset will be created
                     with the given name
        dataset_name (str): the dataset name to use if the dataset could not be found
        input_dir (str): the input directory in S3 for the dataset. If the dataset exists,
                         then the input directory will be checked with this argument
        output_dir (str): the output directory in S3 for the dataset. If the dataset exists,
                          then the output directory will be checked with this argument
        dataset_type (DatasetType, optional): The Lightly Dataset type.
                          Defaults to DatasetType.VIDEOS.
        aws_region (str, optional): The aws region to choose for the
                          lightly buckets. Defaults to "us-west-2".
        metadata (dict, optional): the lightly metadata that will
                           be used for the dataset creation. Defaults to None.
        notify (bool, optional): whether to notify the dataset creation in slack

    Returns:
        str: the dataset id from lightly
    """
    # find existing dataset by id
    dataset = client.get_dataset_by_id(dataset_id) if dataset_id else None
    if dataset:
        logger.info(f"Found existing dataset with content: \n{dataset}")
        client.dataset_id = dataset.id
        return client.dataset_id
    logger.warning("Could not find the dataset. Creating one")
    return create_and_configure_dataset(
        client,
        dataset_name,
        input_dir,
        output_dir,
        dataset_type,
        aws_region,
        metadata,
        notify,
    )
