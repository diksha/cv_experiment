import argparse
import os
import typing
import uuid
from typing import Dict, List, Optional, Tuple

import mergedeep
import sematic
from loguru import logger

from core.infra.sematic.shared.resources import GPU_4CPU_16GB_2x
from core.infra.sematic.shared.utils import PipelineSetup
from core.metaverse.api.queries import (
    get_or_create_task_and_service,
    register_model,
)
from core.ml.common.utils import (
    add_camera_uuid_parser_arguments,
    get_camera_uuids_from_arguments,
)
from core.ml.data.generation.common.pipeline import DatasetMetaData
from core.ml.data.generation.resources.api.dataset_generator import (
    generate_and_register_dataset,
    load_dataset_config,
)
from core.ml.data.generation.resources.api.logset_generator import (
    generate_logset,
    load_logset_config,
)
from core.ml.experiments.tracking.experiment_tracking import (
    ExperimentTracker,
    ExperimentTrackerDetails,
)
from core.ml.training.api.generated_model import GeneratedModel
from core.ml.training.api.training_manager import TrainingManager
from core.structs.dataset import Dataset as VoxelDataset
from core.structs.model import ModelCategory
from core.structs.task import Task, TaskPurpose
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
from core.utils.yaml_jinja import load_yaml_with_jinja


def setup_environment(metaverse_environment: str):
    """Sets up environment variables for training run.

    Args:
        metaverse_environment (str): value of metaverse env
    """
    logger.info("Setting up environment variables")
    os.environ["METAVERSE_ENVIRONMENT"] = metaverse_environment
    os.environ["CLEARML_WEB_HOST"] = get_secret_from_aws_secret_manager(
        "CLEARML_WEB_HOST"
    )
    os.environ["CLEARML_API_HOST"] = get_secret_from_aws_secret_manager(
        "CLEARML_API_HOST"
    )
    os.environ["CLEARML_FILES_HOST"] = get_secret_from_aws_secret_manager(
        "CLEARML_FILES_HOST"
    )
    os.environ["CLEARML_API_ACCESS_KEY"] = get_secret_from_aws_secret_manager(
        "CLEARML_API_ACCESS_KEY"
    )
    os.environ["CLEARML_API_SECRET_KEY"] = get_secret_from_aws_secret_manager(
        "CLEARML_API_SECRET_KEY"
    )


@sematic.func(
    resource_requirements=GPU_4CPU_16GB_2x,
    standalone=True,
)
def train(
    task: Task,
    model_config: typing.Dict[str, object],
    experiment_tracker_details: ExperimentTrackerDetails,
    dataset: VoxelDataset,
    dataset_metadata: DatasetMetaData,
    metaverse_environment: str,
) -> Tuple[str, GeneratedModel]:
    """Train the model

    Args:
        task (Task): task associated to training
        model_config (Dict[str, object]): model configuration
        experiment_tracker_details (ExperimentTrackerDetails): details of experiment tracker
        dataset (VoxelDataset): metaverse registered dataset
        dataset_metadata (DatasetMetaData): metadata of dataset dowloaded
        metaverse_environment (str): environment of metaverse

    Raises:
        Exception: when training is not successful

    Returns:
        GeneratedModel: Model generated
    """

    def upload_model(task: Task, generated_model: GeneratedModel) -> str:
        """
        Uploads the model to S3. The current naming convention is:

        s3://voxel-models/automated/{task.purpose}/{task.uuid}/{uuid}.{model_extension}

        Args:
            task (Task): the current task for the model
            generated_model (GeneratedModel): the model generated from the training framework

        Returns:
            str: the cloud gcs path for the model
        """

        model_extension = os.path.splitext(generated_model.local_model_path)[
            1
        ].lstrip(".")
        bucket = "voxel-models"
        s3_path = (
            f"automated/{task.purpose}/{task.uuid}/"
            f"{str(uuid.uuid4())}.{model_extension}"
        )
        extra_args = {"ContentType": "application/octet-stream"}
        logger.info(f"Uploading model to gcs: {s3_path}")
        upload_file(
            bucket,
            generated_model.local_model_path,
            s3_path,
            extra_args=extra_args,
        )
        return f"s3://{bucket}/{s3_path}"

    setup_environment(metaverse_environment)
    try:
        experiment_tracker = ExperimentTracker(experiment_tracker_details)
        generated_model = (
            TrainingManager(
                model_config=model_config,
                experiment_tracker=experiment_tracker,
            )
            .download_dataset(dataset, dataset_metadata)
            .train(dataset)
        )
        logger.info(f"Status of flushing logs: {experiment_tracker.flush()}")
    except Exception as exception:
        logger.exception("Unable to train")
        raise exception
    s3_path = upload_model(task, generated_model)
    return s3_path, generated_model


def notify_slack(
    task: Task,
    model_metadata: dict,
    model_metrics: dict,
    tracker_details: ExperimentTrackerDetails,
) -> None:
    """
    Notifies slack at the completion of the run with model information
    and any relevant metrics

    Args:
        task (Task): the task used for this job
        model_metadata (dict): the model metadata returned from registering the model
        model_metrics (dict): the model metrics
        tracker_details (ExperimentTrackerDetails): details of tracker

    """
    webhook = SynchronousWebhookWrapper(get_perception_verbose_sync_webhook())
    blocks = [f"*Camera uuids:* {task.camera_uuids}"]
    blocks.extend(
        [
            f"*Experiment Tracking:* {tracker_details}",
            f'*Model UUID:* {model_metadata.get("uuid")}',
            f'*Model Path:* {model_metadata.get("path")}',
        ]
    )

    def prettify(string: str, delimiter: str) -> str:
        """
        Utility to make a string look a little nicer

        Capitalizes all words split by the delimiter

        Args:
            string (str): the original string
            delimiter (str): the delimiter to split

        Returns:
            str: the nicer looking string
        """
        return " ".join([val.capitalize() for val in string.split(delimiter)])

    blocks.extend(
        [
            f"*{prettify(key.replace('/', '_'), '_')}*: {value:.4f}"
            for key, value in model_metrics.items()
        ]
    )
    webhook.post_message_block_with_fields(
        title=f"Model Performance {task.purpose}", blocks=blocks
    )


@sematic.func
def finalize_model(
    task: Task,
    generated_model: GeneratedModel,
    s3_model_path: str,
    training_manager_config: typing.Dict[str, object],
    experiment_config: typing.Dict[str, object],
    tracker_details: List[str],
    dataset: VoxelDataset,
    notify: bool,
) -> typing.Dict[str, object]:
    """Finalize the model

    Args:
        task (Task): task model is for
        generated_model (GeneratedModel): model that is generated
        s3_model_path (str): path of s3
        training_manager_config (typing.Dict[str, object]): config of training manager
        experiment_config (typing.Dict[str, object]): config of experimet
        tracker_details (List[str]): tracker details
        dataset (VoxelDataset): dataset used to generate model
        notify (bool): whether to notify on slack

    Returns:
        typing.Dict[str, object]: _description_
    """

    logger.info("starting to finalize")
    model_metadata = register_model(
        training_manager_config,
        experiment_config,
        tracker_details,
        s3_model_path,
        dataset.uuid,
        generated_model.metrics,
        task,
    )
    logger.success("Registered Model")
    # do some slack hook kind of situation
    if notify:
        notify_slack(
            task, model_metadata, generated_model.metrics, tracker_details
        )
    logger.info(f"Finalized the model {model_metadata}")
    return model_metadata


class ExperimentManager:
    """
    Experiment manager is responsible for managing and book keeping
    registration and tracking for datasets, tasks, and models.

    Responsibilities include:
     1. Creating the task
        a. Registering it in metaverse
     2. Generating the dataset from a config
        a. Registering the dataset
     3. Generating a model from a config
        a. Training the model using the dataset
        b. Evaluating/Testing the model
        c. Registering the model
    """

    def __init__(
        self,
        experiment_config: dict,
        task: Task,
        metaverse_environment: str,
        notify: bool = True,
        override_config: typing.Optional[typing.Dict[str, object]] = None,
    ):
        """
        Initializes the experiment manager

        Args:
            model_config (dict): the config used to generate the model
            dataset_config (dict): the config required to generate the dataset
            logset_config (dict): the config required to generate the logset
            metrics_config (dict): the config required to generate metrics
            override_config (typing.Optional[typing.Dict[str, object]]): the
                        config to overwrite the existing config
        """
        dataset_config_file = experiment_config["dataset_config"]
        logset_config_file = experiment_config["logset_config"]
        model_config_file = experiment_config["model_config"]
        self.experiment_tracker_details = ExperimentTrackerDetails(
            get_project_name(task), str(uuid.uuid4())
        )

        self.tracker = ExperimentTracker(self.experiment_tracker_details)
        self.model_config = load_yaml_with_jinja(model_config_file)
        self.dataset_config_file = dataset_config_file
        self.logset_config_file = logset_config_file
        self.experiment_config = experiment_config
        self.task = task
        self.notify = notify
        self.metaverse_environment = metaverse_environment
        self.override_config = (
            override_config if override_config is not None else {}
        )

    def run(self) -> typing.Dict[str, object]:
        """
        Runs the pipeline end to end.

        1. Generates logset using the logset config file and task
        2. Generates a dataset using a dataset config file, a task
           and a logset
        3. Trains a model using the training manager
            a. Uploads the model to GCS
            b. registers the model in metaverse
        4. Notifies slack with the results of training

        Returns:
            typing.Dict[str, object]: results of the run
        """

        def merge_if_override_exists(
            config: Dict[str, object],
            override: typing.Optional[Dict[str, object]],
        ) -> typing.Dict[str, object]:
            """
            Merges the dictionary and returns it if the override dict is present

            Args:
                config (Dict[str, object]): the configuration file
                override (typing.Optional[Dict[str, object]]): the override dictionary

            Returns:
                typing.Dict[str, object]: the merged/unmerged dictionary
            """
            if override is not None:
                mergedeep.merge(config, override)
            return config

        logset = generate_logset(
            merge_if_override_exists(
                load_logset_config(self.logset_config_file, task=self.task),
                self.override_config.get("logset"),
            )
        )
        # generate dataset config
        (dataset, dataset_metadata) = generate_and_register_dataset(
            merge_if_override_exists(
                load_dataset_config(
                    self.dataset_config_file, logset=logset, task=self.task
                ),
                self.override_config.get("dataset"),
            ),
            logset,
            metaverse_environment=self.metaverse_environment,
        )
        self.model_config = merge_if_override_exists(
            self.model_config, self.override_config.get("model")
        )
        s3_path, generated_model = train(
            self.task,
            self.model_config,
            self.experiment_tracker_details,
            dataset,
            dataset_metadata,
            self.metaverse_environment,
        )
        return finalize_model(
            self.task,
            generated_model,
            s3_path,
            self.model_config,
            self.experiment_config,
            self.tracker.get_details(),
            dataset,
            self.notify,
        )


def parse_args() -> argparse.Namespace:
    """Parses arguments for the binary

    Returns:
        argparse.Namespace: Command line arguments passed in.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config_path",
        type=str,
        help="Path of the configuration defining the task",
    )
    parser.add_argument(
        "--experimenter",
        type=str,
        help="Service account or name of the person running experiments",
        default=os.environ.get("USER", "UNKNOWN"),
    )
    parser.add_argument(
        "--notify",
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to notify slack",
    )
    parser = add_camera_uuid_parser_arguments(parser)
    return parser.parse_args()


def get_project_name(task: Task) -> str:
    """Project name from task and camera_uuid

    Args:
        task (Task): the task for the project

    Returns:
        str: name of the project
    """
    return f"{task.purpose}_{task.uuid}"


@sematic.func
# trunk-ignore(pylint/R0913)
def run_experiment(
    experiment_config_path: str,
    experimenter: str,
    notify: bool,
    camera_uuids: List[str],
    organization: Optional[str],
    location: Optional[str],
    metaverse_environment: str,
    pipeline_setup: PipelineSetup,
    override_config: typing.Optional[typing.Dict[str, object]] = None,
) -> typing.Dict[str, object]:
    """
    Generates an experiment manager and kicks off the experiment manager

    Args:
        experiment_config_path (str): path of the experiment
        experimenter (str): who is experimenting
        notify (bool): whether to notify in slack
        camera_uuids (List[str]): camera uuids to create model for
        organization (Optional[str]): organization to create model for
        location (Optional[str]): location to create model for
        metaverse_environment (str): environment of metaverse
        override_config (typing.Optional[typing.Dict[str, object]]): the
                    config to overwrite the existing config
        pipeline_setup (PipelineSetup): setting up some defaults for pipeline

    Returns:
        typing.Dict[str, object]: experiment results

    Raises:
        Exception: if one is thrown when running the rest of the pipeline
    """
    logger.info(f"Starting training for experimenter {experimenter}")
    setup_environment(metaverse_environment)
    try:
        experiment_configuration: dict = load_yaml_with_jinja(
            experiment_config_path
        )

        camera_uuids: typing.List[str] = get_camera_uuids_from_arguments(
            camera_uuids, organization, location
        )

        task: Task = get_or_create_task_and_service(
            TaskPurpose[experiment_configuration["task_purpose"]],
            ModelCategory[experiment_configuration["service_category"]],
            camera_uuids,
        )

        experiment_manager = ExperimentManager(
            experiment_config=experiment_configuration,
            task=task,
            notify=notify,
            metaverse_environment=metaverse_environment,
            override_config=override_config,
        )
        return experiment_manager.run()
    except Exception as exception:
        logger.exception(exception)
        raise exception
