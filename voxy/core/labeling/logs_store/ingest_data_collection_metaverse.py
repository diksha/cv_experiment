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
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import sematic
from loguru import logger
from sematic.resolvers.silent_resolver import SilentResolver

from core.metaverse.api.data_collection_queries import (
    get_or_create_camera_uuid,
    ingest_data_collection,
)
from core.structs.data_collection import DataCollectionType


@dataclass
class DataCollectionInfo:
    """Represents a data collection to be ingested to Metaverse

    Attributes
    ----------
    data_collection_uuid:
        The unique id of the data collection to be ingested
    is_test:
        Whether the data collection should be used for test or not
    """

    data_collection_uuid: str
    is_test: bool
    data_collection_type: DataCollectionType


@sematic.func
def generate_data_collection_metadata_list(
    data_collection_uuids: List[str],
    is_test: bool,
    data_collection_type: DataCollectionType,
) -> List[DataCollectionInfo]:
    """Generate list of DataCollectionInfo utilizing same test flag and data_collection_type
    Args:
        data_collection_uuids (List[str]): list of data_collection_uuids
        is_test (bool): flag for testing
        data_collection_type (DataCollectionType): DataCollectionType of uuids
    Returns:
        List[DataCollectionInfo]: list of DataCollectionInfo
    """
    return [
        DataCollectionInfo(
            data_collection_uuid=data_collection_uuid,
            is_test=is_test,
            data_collection_type=data_collection_type,
        )
        for data_collection_uuid in data_collection_uuids
    ]


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def ingest_data_collections_to_metaverse(
    data_collections_metadata: List[DataCollectionInfo],
    metaverse_environment: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """# Ingests data collections sent to labeling to metaverse

    ## Parameters
    - **data_collection_metadata**:
        A description of data collection to be ingested. If not provided, information on how
        to ingest the data collection will come from firestore as written to by Symphony or
        Buildkite.
    - **metaverse_environment**:
        The metaverse environment to ingest to. If not set, the value from
        the METAVERSE_ENVIRONMENT will be used. INTERNAL and PROD are some
        example valid values.

    ## Returns
    A tuple where the first element is successfully ingested data collections UUIDs and the
    second is failed ingestion data collection uuids

    ## Raises
    **RuntimeError**: If no source for the data collections to ingest can be identified
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    if metaverse_environment is not None:
        os.environ["METAVERSE_ENVIRONMENT"] = metaverse_environment
    data_collections_failed_to_ingest = []
    data_collections_successful_ingest = []
    for data_collection_metadata in data_collections_metadata:
        camera_uuid = "/".join(
            data_collection_metadata.data_collection_uuid.split("/")[0:4]
        )
        camera_uuid = get_or_create_camera_uuid(camera_uuid)
        result = ingest_data_collection(
            camera_uuid,
            data_collection_metadata.data_collection_uuid,
            data_collection_metadata.is_test,
            data_collection_metadata.data_collection_type,
        )
        if (
            result.errors is not None
            or not result.data["data_collection_create"]["success"]
        ):
            data_collections_failed_to_ingest.append(
                data_collection_metadata.data_collection_uuid
            )
        else:
            data_collections_successful_ingest.append(
                data_collection_metadata.data_collection_uuid
            )
    return (
        data_collections_successful_ingest,
        data_collections_failed_to_ingest,
    )


def extract_data_collection_metadata_for_ingestion(
    is_test: str, local_file_path: str = None
) -> List:
    """Ingests data collections sent to labeling to metaverse.

    Args:
        is_test (bool): Are the data collections ingested for testing
        local_file_path (str): path to a local file containing new
                               line separated uuids of data collections to ingest

    Raises:
        RuntimeError: If no input is not provided

    Returns:
        List of DataCollectionInfo objects.
    """

    def is_test_to_bool(value: Union[str, bool]) -> str:
        """
        Converts a value input to a bool

        Args:
            value (typing.Union[str, bool]): the bool (True/False) is test value
                                            or a string like "True" or "False"

        Returns:
            bool: the is_test value converted to a string
        """
        return value.upper() == "TRUE" if isinstance(value, str) else value

    if local_file_path:
        data_collection_uuids = []
        with open(local_file_path, "r", encoding="UTF-8") as file_ref:
            for data_collection_uuid in file_ref:
                data_collection_uuid = data_collection_uuid.strip("\n").strip()
                data_collection_uuids.append(data_collection_uuid)
        data_collections_metadata = [
            DataCollectionInfo(
                data_collection_uuid=data_collection_uuid,
                is_test=is_test_to_bool(is_test),
                data_collection_type=DataCollectionType.VIDEO,
            )
            for data_collection_uuid in data_collection_uuids
        ]
    else:
        raise RuntimeError("Cannot retrieve data collection uuids")

    return data_collections_metadata


def parse_args() -> argparse.Namespace:
    """Parse arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--local_file_path",
        help="path to a local file containing new \
        line separated uuids of data collections to ingest",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--is_test",
        type=str,
        help="Should data collections be ingested for test or not",
        default="false",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metadata = extract_data_collection_metadata_for_ingestion(
        args.is_test, args.local_file_path
    )
    successful_ingests, failed_ingests = ingest_data_collections_to_metaverse(
        metadata,
        metaverse_environment=os.environ.get(
            "METAVERSE_ENVIRONMENT", "INTERNAL"
        ),
    ).resolve(SilentResolver())

    if len(failed_ingests) > 0:
        logger.warning("Failed to ingest:", "\n".join(failed_ingests))
    else:
        logger.success(
            f"All {len(successful_ingests)} data collections were successfully ingested"
        )
