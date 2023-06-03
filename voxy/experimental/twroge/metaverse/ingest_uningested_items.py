#
# Copyright 2023 Voxel Labs, Inc.
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

from loguru import logger
from sematic import client

from core.infra.sematic.shared.utils import (
    SematicOptions,
    resolve_sematic_future,
)
from core.ml.data.flywheel.lib.dataflywheel import (
    ingest_data_collection_from_collection_input,
)


def get_args() -> argparse.Namespace:
    """
    Gets input arguments

    Returns:
        argparse.Namespace: the parsed args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact_id",
        type=str,
        help="The artifact_id (type: DataCollectionInput) to reingest to metaverse",
        required=True,
    )
    parser.add_argument(
        "--metaverse_environment",
        type=str,
        help="The metaverse environment to ingest to",
        required=True,
    )
    SematicOptions.add_to_parser(parser)
    return parser.parse_args()


def main(
    artifact_id: str, metaverse_environment: str, arguments: argparse.Namespace
):
    """
    Ingest data collections from a DataCollectionInput artifact

    Args:
        artifact_id (str): the artifact id
        metaverse_environment (str): the metaverse environment to ingest to
        arguments (argparse.Namespace): the rest of the commandline args
    """
    # grab the artifact for the data collections input
    # documentation here: https://docs.sematic.dev/diving-deeper/artifacts

    uningested_items = client.get_artifact_value(artifact_id)

    logger.info(f"Ingesting data collections from future: {artifact_id}:")
    logger.info(str(uningested_items)[:1000] + "...")

    future = ingest_data_collection_from_collection_input(
        uningested_items, metaverse_environment=metaverse_environment
    )
    resolve_sematic_future(
        future,
        arguments,
    )


if __name__ == "__main__":
    args = get_args()
    main(args.artifact_id, args.metaverse_environment, args)
