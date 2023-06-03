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
import os
import tempfile

# trunk-ignore(semgrep/python.lang.security.use-defused-xml.use-defused-xml)
# trunk-ignore(bandit/B405)
from xml.etree.ElementTree import Element, ElementTree, fromstring, tostring

from loguru import logger

from core.infra.cloud.gcs_utils import (
    get_files_in_bucket,
    read_from_gcs,
    upload_to_gcs,
)
from core.labeling.cvat.client import CVATClient
from core.utils.aws_utils import get_secret_from_aws_secret_manager

CVAT_HOST = "cvat.voxelplatform.com"


def main():
    credentials = get_secret_from_aws_secret_manager("CVAT_CREDENTIALS")
    cvat_client = CVATClient(CVAT_HOST, credentials, project_id=10)
    # trunk-ignore(pylint/R1702)
    for camera_config_file in get_files_in_bucket(
        "voxel-raw-labels", prefix="camera_config"
    ):
        full_gcs_path = "gs://{}/{}".format(
            "voxel-raw-labels", camera_config_file.name
        )
        # trunk-ignore(bandit/B314)
        tree = ElementTree(fromstring(read_from_gcs(full_gcs_path)))
        root = tree.getroot()
        door_id = 2
        attribute_added = False
        for child in list(root):
            if child.tag == "track":
                if child.attrib["label"] == "door":
                    door_changed = False
                    for polygon_child in child:
                        flag = 0
                        for attrib_child in polygon_child:
                            if attrib_child.attrib["name"] == "door_id":
                                flag = 1
                                break
                        if flag == 0:
                            attribute_added = True
                            door_changed = True
                            new_attrib = Element("attribute")
                            new_attrib.attrib["name"] = "door_id"
                            new_attrib.text = str(door_id)
                            polygon_child.append(new_attrib)
                if door_changed:
                    door_id = door_id + 1

        if attribute_added:
            with tempfile.NamedTemporaryFile() as temp:
                with open(temp.name, "w") as label_file:
                    label_file.write(
                        tostring(root, method="xml").decode("utf-8")
                    )
                cvat_client.upload_cvat_label(
                    os.path.splitext(camera_config_file.name)[0],
                    "cvat",
                    temp.name,
                )
                upload_to_gcs(
                    f"gs://voxel-raw-labels/{camera_config_file.name}",
                    temp.name,
                )
                logger.info(f"Added for {camera_config_file.name}")
        else:
            logger.info(f"Not added for {camera_config_file.name}")


if __name__ == "__main__":
    main()
