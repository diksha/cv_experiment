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
import uuid
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import fromstring

from google.cloud import storage

from core.labeling.label_store.labels_backend import LabelsBackend
from core.labeling.video_helper_mixin import VideoHelperMixin


class AddDoors(VideoHelperMixin):
    def __init__(self, video_uuid):
        print(video_uuid)
        self.video_uuid = video_uuid
        self.label_reader = LabelsBackend(
            consumable=False, label_format="xml"
        ).label_reader()
        self.tree = ElementTree(fromstring(self.label_reader.read(self.video_uuid)))
        self.root = self.tree.getroot()
        self.doors = json.loads(
            open(
                os.path.join(
                    os.environ["BUILD_WORKSPACE_DIRECTORY"],
                    "data",
                    "artifacts",
                    "doors.json",
                )
            ).read()
        )["CAMERA_DOOR_POLYGONS"]

    def add_door_label_spec(self, meta_element):
        door_label = Element("label")
        name = Element("name")
        name.text = "DOOR"
        attributes = Element("attributes")
        door_label.append(name)

        attribute = Element("attribute")
        name = Element("name")
        name.text = "door_id"
        mutable = Element("mutable")
        mutable.text = "False"
        default_value = Element("default_value")
        default_value.text = "None"
        input_type = Element("input_type")
        input_type.text = "text"
        attribute.extend([name, mutable, default_value, input_type])
        attributes.append(attribute)

        attribute = Element("attribute")
        name = Element("name")
        name.text = "open"
        mutable = Element("mutable")
        mutable.text = "True"
        default_value = Element("default_value")
        default_value.text = "False"
        input_type = Element("input_type")
        input_type.text = "checkbox"
        attribute.extend([name, mutable, default_value, input_type])
        attributes.append(attribute)

        door_label.append(attributes)

        for child in meta_element:
            if child.tag == "task":
                for sub_child in child:
                    if sub_child.tag == "labels":
                        sub_child.append(door_label)

    def add_doors(self, num_frames):
        track = Element("track")
        track.attrib["id"] = "None"
        track.attrib["label"] = "DOOR"

        door_id_attribute = Element("attribute", {"name": "door_id"})
        door_id_attribute.text = "1"

        open_attribute = Element("attribute", {"name": "open"})
        open_attribute.text = "False"

        for i in range(num_frames):
            polygon = Element("polygon")
            polygon.attrib["frame"] = str(i)
            polygon.attrib[
                "points"
            ] = "426.98,0.00;652.00,0.00;645.00,179.00;448.00,175.00"
            polygon.attrib["outside"] = "0"
            polygon.attrib["occluded"] = "0"
            polygon.attrib["keyframe"] = "1"
            polygon.extend([door_id_attribute, open_attribute])
            track.append(polygon)
        self.root.append(track)

    def parse(self):
        for child in self.root:
            if child.tag == "meta":
                self.add_door_label_spec(child)
        num_frames = 10
        self.add_doors(num_frames)
        self.tree.write("/home/anurag_voxelsafety_com/voxel/test.xml")


if __name__ == "__main__":
    video_uuid = "americold/modesto/e_dock_north/ch12/e_dock_north_ch12_20201102001829_20201102041946_lower_1201_upper_1227~f10"
    output_file_path = "gs://{}.json".format(video_uuid)
    convertor = AddDoors(video_uuid)
    convertor.parse()
