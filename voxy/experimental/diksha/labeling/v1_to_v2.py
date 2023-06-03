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
import tempfile
from xml.etree.ElementTree import ElementTree, fromstring, tostring

from core.labeling.cvat.client import CVATClient
from core.labeling.label_store.labels_backend import LabelsBackend
from core.utils.aws_utils import get_secret_from_aws_secret_manager

CVAT_HOST = "cvat.voxelplatform.com"


class V1ToV2LabelConvert:
    """Convert v1 pit and person to v2 pit and person category"""

    def __init__(self, video_uuid, ip_filepath=None, output_file_path=None):
        credentials = get_secret_from_aws_secret_manager("CVAT_CREDENTIALS")
        self.cvat_client = CVATClient(CVAT_HOST, credentials)
        self.video_uuid = video_uuid
        labels_backend = LabelsBackend(consumable=False, label_format="xml")
        self.label_reader = labels_backend.label_reader()
        self.tree = ElementTree(
            fromstring(self.label_reader.read(self.video_uuid))
        )
        self.root = self.tree.getroot()

    def parse(self):
        """Parse

        Raises:
            RuntimeError: _description_
        """
        for child in list(self.root):
            if child.tag == "track" and (
                child.attrib["label"] == "PERSON_V2"
                or child.attrib["label"] == "PIT_V2"
            ):
                raise RuntimeError("Person pit v2 already exists")
        for child in list(self.root):
            if child.tag == "track":
                if child.attrib["label"] == "PERSON":
                    child.attrib["label"] = "PERSON_V2"
                if child.attrib["label"] == "PIT":
                    child.attrib["label"] = "PIT_V2"
        with tempfile.NamedTemporaryFile() as temp:
            with open(temp.name, "w") as label_file:
                label_file.write(
                    tostring(self.root, method="xml").decode("utf-8")
                )
                self.cvat_client.upload_cvat_label(
                    self.video_uuid, "cvat", temp.name
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--videos",
        metavar="V",
        type=str,
        nargs="+",
        help="video uuid(s) to ingest",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for video in args.videos:
        try:
            v1_to_v2_label_convert = V1ToV2LabelConvert(video)
            v1_to_v2_label_convert.parse()
            print(f"done {video}")
        except Exception as e:
            print(f"not done {video} {e}")
