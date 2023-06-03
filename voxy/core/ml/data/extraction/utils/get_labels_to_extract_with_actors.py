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

import argparse
import random
from typing import List, Optional

import sematic
from loguru import logger

from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.metaverse.metaverse import Metaverse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--actors_to_keep",
        nargs="+",
        type=str,
        required=True,
        help="Inclusive list of actor categories data collections must have",
    )
    return parser.parse_args()


def get_data_collection_uuids_containing_actor_categories(
    metaverse: Metaverse,
    categories: List[str],
) -> List[str]:
    """Get data collection uuids for actor category

    Args:
        metaverse (Metaverse): metaverse instance
        categories (List[str]): actor category

    Returns:
        List[str]: list of data collection uuids
    """
    query = """query get_data_collection_contains_actor_categories($categories: [String!]) {
        data_collection_contains_actor_categories(categories: $categories, data_collection_types: "VIDEO") {
            voxel_uuid
        }
    }
    """
    qvars = {"categories": categories}
    data_collection_result = metaverse.schema.execute(query, variables=qvars)
    if data_collection_result.errors:
        logger.error(
            f"Could not find data collection: {data_collection_result.errors}"
        )
    else:
        logger.info(
            (
                f"Data collection result"
                f' {data_collection_result.data["data_collection_contains_actor_categories"]}'
            )
        )
    data_collection_uuid_list = []
    for i, _ in enumerate(
        data_collection_result.data[
            "data_collection_contains_actor_categories"
        ]
    ):
        voxel_uuid = data_collection_result.data[
            "data_collection_contains_actor_categories"
        ][i]["voxel_uuid"]
        if voxel_uuid:
            data_collection_uuid_list.append(voxel_uuid)
    return data_collection_uuid_list


@sematic.func(resource_requirements=CPU_1CORE_4GB, standalone=True)
def get_labels_to_extract_with_actors(
    actors_to_keep: List[str],
    metaverse_environment: Optional[str] = None,
    sample_frac: Optional[float] = None,
) -> List[str]:
    """Get list of data collections to extract that have actors in actors_to_keep.
    Optionally subsample the data for testing.

    Args:
        actors_to_keep (List[str]): list of actor category names to find labels for
        metaverse_environment (Optional[str]): Metaverse environment to use
        sample_frac (Optional[float], optional): Fraction of data to sample.
            Defaults to None (use all data).

    Returns:
        List[str]: List of data collection UUIDs with appopriate actors
    """
    metaverse = Metaverse(environment=metaverse_environment)
    data_collection_uuids = (
        get_data_collection_uuids_containing_actor_categories(
            metaverse, actors_to_keep
        )
    )
    if sample_frac is not None:
        data_collection_uuids = random.sample(
            data_collection_uuids,
            int(len(data_collection_uuids) * sample_frac),
        )
        logger.info(
            f"get_labels_to_extract_with_actor sampled {len(data_collection_uuids)} items"
        )
    return data_collection_uuids
