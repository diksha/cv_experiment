#
# Copyright 2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import os
from dataclasses import dataclass
from typing import List, Optional

import sematic

from core.labeling.logs_store.ingest_image_collection import (
    ImagesSource,
    ingest_image_collection,
)
from core.labeling.scale.runners.create_scale_tasks import (
    ScaleTaskSummary,
    create_scale_tasks,
)
from core.metaverse.api.datapool_queries import (
    get_or_create_datapool_from_task,
    update_datapool,
)
from core.ml.common.utils import get_merged_config
from core.ml.data.collection.data_collector import (
    Incident,
    IncidentsFromPortalInput,
    select_incidents_from_portal,
)
from core.ml.data.curation.lib.lightly_worker import (
    LightlyVideoFrameSequence,
    run_lightly_worker,
)
from core.structs.frame import Frame
from core.structs.task import Task
from core.utils.logging.slack.get_slack_webhooks import (
    get_perception_verbose_sync_webhook,
)
from core.utils.logging.slack.synchronous_webhook_wrapper import (
    SynchronousWebhookWrapper,
)
from core.utils.yaml_jinja import load_yaml_with_jinja


@dataclass
class DataCollectionInput:
    """A summary of the frames that Lightly identified for labeling from a video

    Has fields exactly matching those in a sequence_information.json file from Lightly.

    See here for relevant Lightly docs:
    https://github.com/lightly-ai/lightly/blob/11c5c2b0f22a688c11258f98c32a2bcefc6c755b/docs/source/docker/advanced/sequence_selection.rst

    Attributes
    ----------
    data_collection_name:
        The name of the data collection that was checked for labeling
    camera_uuid:
        uuid of the camera for data collection
    frames:
        A list of the frames for the images pulled from the identified frames
    data_dir:
        directory where the files exist
    is_test:
        Whether video is ingested for test
    """

    data_collection_name: str
    camera_uuid: str
    frames: List[Frame]
    data_dir: str
    is_test: bool


@dataclass
class DataFlywheelSummary:
    scale_task_summary: ScaleTaskSummary
    did_update_datapool: bool


class DataFlywheel:
    """
    Dataflywheel.

    Responsibilities include:
        1. Grab the existing datapool or create it
        2. Grab the data from the dataflywheel collector from portal
        3. Run lightly curation on the data from portal incidents
    """

    BASE_DATA_DATAFLYWHEEL_CONFIG_DIRECTORY = (
        "core/infra/sematic/perception/data_ingestion/dataflywheel/configs"
    )

    def __init__(
        self,
        task: Task,
        start_date: str,
        end_date: str,
        max_incidents: int,
        notify: bool = True,
        overwrite_config_file: Optional[str] = None,
    ):
        """
        Initializes dataflywheel. Loads configs from:

        BASE_DATA_COLLECTOR_DIRECTORY
        BASE_LIGHTLY_CONFIG_DIRECTORY

        Args:
            task (Task): the task to run the dataflywheel for (e.g. Doors on camera:
                         `americold/modesto/0011/cha`)
            notify (bool, optional): whether or not to notify via slack. Defaults to True.
            overwrite_config_file (Optional[str]): config that will overwrite dataflywheel configs

        """
        self.task = task
        self.notify = notify
        task_purpose_name = self.task.purpose
        self.max_incidents = max_incidents

        # get the dataflywheel configuration
        self.dataflywheel_config = load_yaml_with_jinja(
            os.path.join(
                self.BASE_DATA_DATAFLYWHEEL_CONFIG_DIRECTORY,
                f"{task_purpose_name}.yaml",
            )
        )
        overwrite_config = (
            load_yaml_with_jinja(overwrite_config_file)
            if overwrite_config_file
            else {}
        )
        self.data_collector_config = get_merged_config(
            load_yaml_with_jinja(
                self.dataflywheel_config["data_collection_config_file"],
                task=task.to_dict(),
            ),
            overwrite_config.get("data_collector_config", {}),
        )
        self.lightly_config = get_merged_config(
            load_yaml_with_jinja(
                self.dataflywheel_config["lightly_config_file"]
            ),
            overwrite_config.get("lightly_config", {}),
        )
        self.ingestion_config = get_merged_config(
            load_yaml_with_jinja(
                self.dataflywheel_config["ingestion_config_file"]
            ),
            overwrite_config.get("ingestion_config", {}),
        )
        self.start_date = start_date
        self.end_date = end_date

    def notify_slack(self, task: Task) -> None:
        """
        Notifies slack at the completion of the run with the dataflywheel
        and any relevant metrics

        Args:
            task (Task): the task used for this job

        """
        webhook = SynchronousWebhookWrapper(
            get_perception_verbose_sync_webhook()
        )
        blocks = []
        webhook.post_message_block_with_fields(
            title=f"DataFlywheel: {task.purpose}", blocks=blocks
        )

    def run(self, metaverse_environment: Optional[str]) -> DataFlywheelSummary:
        """
        Runs the dataflywheel pipeline end to end:
        1. Grabs the existing datapool or creates it
        2. Grabs the data from the dataflywheel collector from portal
        3. Runs lightly curation on the dataflywheel collector

        Args:
            metaverse_environment (Optional[str]): metaverse environment to use
        Returns:
            DataFlywheelSummary: summary containing scale tasks and if the
                corresponding datapool was updated

        """

        datapool = get_or_create_datapool_from_task(
            self.task,
            self.lightly_config,
            metaverse_environment=metaverse_environment,
        )

        incidents = select_incidents_from_portal(
            IncidentsFromPortalInput(
                config=self.data_collector_config,
                start_date=self.start_date,
                end_date=self.end_date,
                output_bucket="voxel-lightly-input",
                output_path=datapool.input_directory,  # trunk-ignore(pylint/E1101)
                max_num_incidents=self.max_incidents,
                metaverse_environment=metaverse_environment,
            )
        )
        lightly_output = run_lightly_worker(
            dataset_id=datapool.lightly_uuid,  # trunk-ignore(pylint/E1101)
            dataset_name=datapool.name,  # trunk-ignore(pylint/E1101)
            input_directory=datapool.input_directory,  # trunk-ignore(pylint/E1101)
            output_directory=datapool.output_directory,  # trunk-ignore(pylint/E1101)
            config=self.lightly_config,
            dataset_type=datapool.dataset_type,  # trunk-ignore(pylint/E1101)
            notify=self.notify,
        )

        data_collection_inputs = generate_data_collections(
            lightly_output,
            incidents,
            datapool.output_directory,  # trunk-ignore(pylint/E1101)
            datapool.ingested_data_collections,  # trunk-ignore(pylint/E1101)
        )
        ingested_data_collections = (
            ingest_data_collection_from_collection_input(
                data_collection_inputs,
                metaverse_environment=metaverse_environment,
            )
        )
        scale_task_summary = create_scale_tasks(
            ingested_data_collections, prefix="", **self.ingestion_config
        )
        flywheel_summary = generate_flywheel_summary_and_update_datapool(
            scale_task_summary,
            datapool.uuid,  # trunk-ignore(pylint/E1101)
            ingested_data_collections,
            metaverse_environment=metaverse_environment,
        )
        return flywheel_summary


@sematic.func
def generate_data_collections(
    lightly_output: List[LightlyVideoFrameSequence],
    incidents: List[Incident],
    data_directory: str,
    ingested_data_collections: Optional[List[str]],
) -> List[DataCollectionInput]:
    """Sematified data collection generation
    Args:
        lightly_output (List[LightlyVideoFrameSequence]): list of video frame sequences
            from lightly
        incidents (List[Incident]): list of incidents from data collector
        data_directory (str): directory data exists in
        ingested_data_collections (Optional[List[str]]): data collections already ingested
    Returns:
        List[DataCollectionInput]: DataCollectionInput list to be used to create tasks
    """
    data_collections = []
    incident_uuid_map = {}
    if not ingested_data_collections:
        ingested_data_collections = []
    for incident in incidents:
        incident_uuid_map[incident.incident_uuid] = incident
    lightly_output = [
        datacollection
        for datacollection in lightly_output
        if incident_uuid_map.get(datacollection.video_name, None)
        and (
            os.path.join(
                incident_uuid_map.get(
                    datacollection.video_name, None
                ).camera_uuid,
                datacollection.video_name,
            )
            not in ingested_data_collections
        )
    ]
    for datacollection in lightly_output:
        incident = incident_uuid_map.get(datacollection.video_name, None)
        data_collections.append(
            DataCollectionInput(
                data_collection_name=datacollection.video_name,
                camera_uuid=incident.camera_uuid,
                frames=[
                    Frame.from_dict({"relative_image_path": frame})
                    for frame in datacollection.frame_names
                ],
                data_dir=os.path.join(
                    "s3://voxel-lightly-output",
                    data_directory,
                    ".lightly/frames",
                ),
                is_test=False,
            )
        )

    return data_collections


@sematic.func
def ingest_data_collection_from_collection_input(
    data_collections: List[DataCollectionInput],
    metaverse_environment: Optional[str] = None,
) -> List[str]:
    """Ingest data collection to voxel-logs and metaverse

    Args:
        data_collections (List): List of data collections input
        metaverse_environment (Optional[str]): Metaverse environment for
            ingestion

    Returns:
        List: datacollections created
    """
    ingested_data_collections = []
    for data_collection in data_collections:
        ingested_data_collections.append(
            ingest_image_collection(
                output_folder=data_collection.camera_uuid,
                src=ImagesSource.S3,
                images_path=[
                    os.path.join(
                        data_collection.data_dir, frame.relative_image_path
                    )
                    for frame in data_collection.frames
                ],
                image_collection_name=data_collection.data_collection_name,
                is_test=data_collection.is_test,
                metaverse_environment=metaverse_environment,
            )
        )
    return ingested_data_collections


@sematic.func
def generate_flywheel_summary_and_update_datapool(
    scale_task_summary: ScaleTaskSummary,
    datapool_uuid: str,
    ingested_data_collections: List[str],
    metaverse_environment: Optional[str] = None,
) -> DataFlywheelSummary:
    """Generate output for dataflywheel and update datapool
    Args:
        scale_task_summary (List[ScaleTaskSummary]): list of scale task ingestion summary
        datapool_uuid (str): lightly datapool uuid to update
        ingested_data_collections (List[str]): list of ingested videos
        metaverse_environment (Optional[str]): metaverse environment to update datapool
    Returns:
        DataFlywheelSummary: Summary of dataflywheel run
    """
    did_update_datapool = update_datapool(
        datapool_uuid,
        ingested_data_collections,
        metaverse_environment=metaverse_environment,
    )
    return DataFlywheelSummary(
        scale_task_summary=scale_task_summary,
        did_update_datapool=did_update_datapool,
    )
