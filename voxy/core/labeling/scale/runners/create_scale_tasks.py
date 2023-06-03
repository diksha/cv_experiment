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
import argparse
from dataclasses import dataclass
from typing import List

import sematic
from loguru import logger
from sematic.resolvers.silent_resolver import SilentResolver

import core.labeling.scale.registry.register_components  # trunk-ignore(pylint/W0611,flake8/F401)
from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.labeling.scale.lib.scale_batch_wrapper import ScaleBatchWrapper
from core.labeling.scale.registry.registry import ScaleTaskCreatorRegistry
from core.structs.data_collection import DataCollectionType


@dataclass
class ScaleTaskSummary:
    """A summary of the labelling tasks created for Scale

    Attributes
    ----------
    video_uuids:
        The UUIDs of the videos for which labelling was requested
    fps:
        The frame-rate of the requested videos
    task_type:
        The labelling task type that was requested
    batch_name:
        Batch name for the videos that were ingested
    task_unique_ids:
        Unique ids of the tasks that were created
    """

    video_uuids: List[str]
    fps: int
    task_type: str
    data_type: DataCollectionType
    batch_name: str
    task_unique_ids: List[str]
    generate_hypothesis: bool


# trunk-ignore-begin(pylint/W9015,pylint/W9011,pylint/W9006)
@sematic.func(
    resource_requirements=CPU_1CORE_4GB,
    standalone=True,
)
def create_scale_tasks(
    video_uuids: List[str],
    fps: float,
    task_type: str,
    prefix: str,
    credentials_arn: str,
    dry_run: bool = False,
    generate_hypothesis: bool = False,
) -> ScaleTaskSummary:
    """# Create Scale.ai labelling tasks

    ## Parameters
    - **video_uuids**:
        The UUIDs for the videos tasks should be created for. If passed as an empty
        list, the UUIDS will be pulled from the output store or the symphony collection.
    - **fps**:
        The frames-per-second of the videos for labelling
    - **task_type**:
        The kind of Scale.ai labelling task to create. Should be one of the
        classes registered under the ScaleTaskCreatorRegistry
    - **prefix**:
        The prefix for the batch of Scale tasks to create
    - **credentials_arn**:
        The credentials to access the scale api (optional)

    ## Returns
    A summary of what tasks were created

    ## Raises
    Exception if raised by "create_task" call
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011,pylint/W9006)
    task_uuids = []
    if generate_hypothesis:
        prefix = f"{prefix}_hypothesis"
    task_creation_object = ScaleTaskCreatorRegistry.get_instance(
        task_type,
        {
            "batch_name_prefix": prefix,
            "credentials_arn": credentials_arn,
            "dry_run": dry_run,
        },
    )
    try:
        for video_uuid in video_uuids:
            task_uuids.extend(
                task_creation_object.create_task(
                    video_uuid,
                    fps,
                    generate_hypothesis=generate_hypothesis,
                )
            )
    except Exception as e:
        logger.exception("Task creation failed")
        ScaleBatchWrapper(credentials_arn).cancel_batch(
            task_creation_object.batch.name
        )
        # Finalize the batch to not leave in hanging state.
        task_creation_object.finalize()
        raise e

    task_creation_object.finalize()
    scale_task_summary = ScaleTaskSummary(
        video_uuids=video_uuids,
        fps=fps,
        task_type=task_type,
        data_type=task_creation_object.get_data_collection_type(),
        batch_name=task_creation_object.batch.name,
        task_unique_ids=task_uuids,
        generate_hypothesis=generate_hypothesis,
    )
    logger.info(f"Scale tasks created {scale_task_summary}")
    return scale_task_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--fps",
        metavar="F",
        type=float,
        required=False,
        default=0,
        help="desired frames per second",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="VideoPlaybackAnnotationTask",
        help="Type of task",
    )
    parser.add_argument(
        "-v",
        "--videos",
        metavar="V",
        type=str,
        nargs="+",
        help="video uuid(s) to ingest",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="batch name prefix",
    )
    parser.add_argument(
        "-a",
        "--credentials_arn",
        type=str,
        default=(
            "arn:aws:secretsmanager:us-west-2:203670452561:"
            "secret:scale_credentials-WHUbar"
        ),
        help="Credetials arn",
    )
    parser.add_argument(
        "--generate_hypothesis",
        action="store_true",
        help="Generate hypotheses for the task",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    videos = args.videos if args.videos is not None else []
    logger.info(args.generate_hypothesis)
    create_scale_tasks(
        videos,
        args.fps,
        args.type,
        args.prefix,
        args.credentials_arn,
        generate_hypothesis=args.generate_hypothesis,
    ).resolve(SilentResolver())
