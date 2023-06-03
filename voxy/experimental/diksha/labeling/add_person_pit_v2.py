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
from copy import deepcopy
from xml.etree.ElementTree import ElementTree, fromstring, tostring

from core.labeling.cvat.client import CVATClient
from core.labeling.label_store.labels_backend import LabelsBackend
from core.utils.aws_utils import get_secret_from_aws_secret_manager

CVAT_HOST = "cvat.voxelplatform.com"


class LabelAppend:
    """Adds v2 labels for person and pit"""

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

    def parse_track(self, track_element_):
        track_element = deepcopy(track_element_)
        track_id = track_element.attrib["id"]
        track_element.attrib.pop("id")
        if track_element.attrib["label"] == "PERSON":
            track_element.attrib["label"] = "PERSON_V2"
        if track_element.attrib["label"] == "PIT":
            track_element.attrib["label"] = "PIT_V2"
        for box in list(track_element):
            new_attr = deepcopy(box[0])
            new_attr.attrib["name"] = "track_id"
            new_attr.text = str(track_id)
            box.append(new_attr)
        return track_element

    def parse(self):
        newTreeMac = deepcopy(self.tree)
        newTree = newTreeMac.getroot()
        for child in list(newTree):
            if child.tag == "track" and (
                child.attrib["label"] == "PERSON_V2"
                or child.attrib["label"] == "PIT_V2"
            ):
                raise RuntimeError("Person pit v2 already exists")
        for child in list(self.root):
            if child.tag == "track" and (
                child.attrib["label"] == "PERSON"
                or child.attrib["label"] == "PIT"
            ):
                newTree.append(self.parse_track(child))
        with tempfile.NamedTemporaryFile() as temp:
            with open(temp.name, "w") as label_file:
                label_file.write(
                    tostring(newTree, method="xml").decode("utf-8")
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
            label_append = LabelAppend(video)
            label_append.parse()
            print(f"done {video}")
        except Exception as e:
            print(f"not done {video} {e}")
