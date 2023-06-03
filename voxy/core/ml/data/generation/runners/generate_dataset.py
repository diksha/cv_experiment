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
import uuid

from loguru import logger
from sematic.resolvers.silent_resolver import SilentResolver

from core.ml.common.utils import (
    add_camera_uuid_parser_arguments,
    get_camera_uuids_from_arguments,
)
from core.ml.data.generation.resources.api.dataset_generator import (
    generate_and_register_dataset,
    load_dataset_config,
)
from core.ml.data.generation.resources.api.logset_generator import (
    generate_logset,
    load_logset_config,
)
from core.structs.task import Task, TaskPurpose


def parse_args() -> argparse.Namespace:
    """
    Parse the arguments from commandline

    Returns:
        argparse.Namespace: the namespace object for known commanline arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_config",
        required=True,
        type=str,
        help="The path of the config",
    )
    parser.add_argument(
        "--logset_config",
        required=True,
        type=str,
        help="The path of the config",
    )
    parser = add_camera_uuid_parser_arguments(parser)
    return parser.parse_known_args()[0]


def main(args: argparse.Namespace):
    """
    The main entry point for the dataset generation script. This takes in the
    input commandline yaml file, generates a pipeline and runs the pipeline

    Args:
        args (argparse.Namespace): the arguments required to construct pipeline
                                   (sources, transforms, readers, writers)
    Raises:
        Exception: raised when the dataset generation fails. A full stack trace is
                   provided to help debug the issue
    """
    try:
        task = Task(
            uuid=str(uuid.uuid4()),
            purpose=TaskPurpose.UNKNOWN,
            metadata={},
            camera_uuids=get_camera_uuids_from_arguments(
                args.camera_uuids, args.organization, args.location
            ),
            service_id=str(uuid.uuid4()),
        )
        logset = generate_logset(load_logset_config(args.logset_config, task))
        generate_and_register_dataset(  # trunk-ignore(pylint/E1101)
            load_dataset_config(args.dataset_config, logset=logset, task=task),
            logset=logset,
        ).resolve(SilentResolver())
    except Exception as exception:
        logger.exception(exception)
        raise exception


if __name__ == "__main__":
    main(parse_args())
