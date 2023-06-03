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
import json
import os

import yaml
from loguru import logger

from core.infra.cloud import gcs_utils
from core.labeling.label_conversion.read_consumable_labels import (
    curate_data_collection_metadata,
    get_uuid,
    read,
)
from core.metaverse.api.data_collection_queries import (
    add_data_collection_labels,
)
from core.metaverse.metaverse import Metaverse
from core.structs.data_collection import DataCollectionType
from core.structs.video import Video


def main():
    metaverse = Metaverse(environment="INTERNAL")
    dictionary = {}
    for r, _, f in os.walk(
        "/home/diksha_voxelsafety_com/voxel/data/scenario_sets"
    ):
        for file in f:
            path = os.path.join(r, file)
            if (
                "regression_scenarios.yaml" not in file
                and ".yaml" in file
                and "temporary" not in path
            ):
                with open(path, "r") as stream:
                    dictionary = yaml.safe_load(stream)
                    for scenario in dictionary["scenarios"]:
                        video_uuid = scenario["video_uuid"]
                        if scenario.get("incidents") is not None:
                            incidents = scenario.get("incidents")
                            print(incidents)
                            if not gcs_utils.does_gcs_blob_exists(
                                "gs://voxel-consumable-labels/"
                                + "v1/"
                                + video_uuid
                                + ".json"
                            ):
                                pass
                                splits = video_uuid.split("/")
                                camera_uuid = get_uuid(
                                    splits[0],
                                    splits[1],
                                    splits[2],
                                    splits[3],
                                    metaverse,
                                )
                                data_metadata = {}
                                data_metadata["path"] = (
                                    "gs://voxel-logs/" + video_uuid + ".mp4"
                                )
                                data_metadata["violations"] = [
                                    {"version": "v1", "violations": incidents}
                                ]
                                data_metadata["name"] = video_uuid
                                data_metadata["is_test"] = False
                                data_metadata["camera_uuid"] = camera_uuid
                                data_metadata = (
                                    curate_data_collection_metadata(
                                        data_metadata
                                    )[1:-1]
                                )
                                result = add_data_collection_labels(
                                    data_metadata,
                                    None,
                                    DataCollectionType.VIDEO,
                                )
                                logger.info(
                                    f"Added scenario to metaverse with result {result}"
                                )
                            else:
                                for (
                                    label_file
                                ) in gcs_utils.get_files_in_bucket(
                                    "voxel-consumable-labels",
                                    prefix="v1/" + video_uuid,
                                ):
                                    video_uuid = os.path.splitext(
                                        label_file.name.replace("v1/", "")
                                    )[0]
                                    video = Video.from_dict(
                                        json.loads(
                                            label_file.download_as_string()
                                        )
                                    )
                                    read(
                                        video,
                                        video_uuid,
                                        incidents,
                                    )


if __name__ == "__main__":
    main()
