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
import collections
import csv
import json
import os
import uuid
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import fromstring

from core.labeling.video_helper_mixin import VideoHelperMixin
from core.structs.attributes import Point

CsvRow = collections.namedtuple(
    "CsvRow", "video_uri label instance_id time_offset xtl ytl xtr ytr xbr ybr xbl ybl"
)


class CvatAutoMLConveter(VideoHelperMixin):
    """Converts CVAT XML annotation data to Cloud Auto ML Dataset.

    Each XML file corresponds to one video file.

    General structure of CVAT XML is:
        <root>
            <meta/>
            <track id="5" label="Person" group_id="1">
                <box frame="0" outside="0" occluded="0" keyframe="1" xtl="1022.94" ytl="746.60" xbr="1155.66" ybr="858.74">
                    <attribute name="Occluded">true</attribute>
                    <attribute name="Truncated">false</attribute>
                </box>
                <box>...</box>
                <box>...</box>
            </track>
            <track>...</track>
            <track>...</track>
        </root>

    The Dataset is generated as csv with the format:
    video_uri, label, instance_id, time_offset, x_relative_min, y_relative_min, x_relative_max,y_relative_min,
    x_relative_max,y_relative_max, x_relative_min, y_relative_max
    """

    def __init__(
        self,
        video_uuid,
        input_labels_path,
        output_labels_path,
        video_folder_path="gs://voxel-videos/",
    ):
        self.tree = ElementTree(ElementTree.parse(self, source=input_labels_path))
        self.video_uuid = video_uuid
        self.root = self.tree.getroot()
        self.frame_timestamp_ms_map = {}
        self.labels = []
        self.video_folder_path = video_folder_path
        self.output_labels_path = output_labels_path
        self.timestamp_round_decimals = 4

    def parse_box(self, box_element, track_id):
        if not self.frame_height or not self.frame_width:
            raise (
                "Height and Width of original frame unknown. Cannot generate dataset without this."
            )
        frame_number = box_element.attrib["frame"]
        relative_timestamp_ms = self.frame_timestamp_ms_map[int(frame_number)]
        relative_timestamp_s = float(
            round(relative_timestamp_ms / 1000, self.timestamp_round_decimals)
        )

        top_left = Point(
            float(box_element.attrib["xtl"]), float(box_element.attrib["ytl"])
        )
        bottom_right = Point(
            float(box_element.attrib["xbr"]), float(box_element.attrib["ybr"])
        )
        top_right = Point(
            float(box_element.attrib["xbr"]), float(box_element.attrib["ytl"])
        )
        bottom_left = Point(
            float(box_element.attrib["xtl"]), float(box_element.attrib["ybr"])
        )

        is_forklift = box_element.findall(".//*[@name='Forklift']")
        # Add only forklift labels to the dataset.
        if is_forklift:
            # Normalize the vertices before storing
            self.labels.append(
                CsvRow(
                    video_uri=os.path.join(
                        self.video_folder_path, self.video_uuid + ".mp4"
                    ),
                    label="forklift",
                    instance_id=track_id,
                    time_offset=relative_timestamp_s,
                    xtl=top_left.x / self.frame_width,
                    ytl=top_left.y / self.frame_height,
                    xtr=top_right.x / self.frame_width,
                    ytr=top_right.y / self.frame_height,
                    xbr=bottom_right.x / self.frame_width,
                    ybr=bottom_right.y / self.frame_height,
                    xbl=bottom_left.x / self.frame_width,
                    ybl=bottom_left.y / self.frame_height,
                )
            )

    def parse_track(self, track_element):
        track_id = int(track_element.attrib["id"])
        for child in track_element:
            if child.tag == "box":
                self.parse_box(child, track_id)
            else:
                print(
                    "Unsupported tag type found during parse_track: {}".format(
                        child.tag
                    )
                )

    def parse_meta(self, meta_element):
        task_element = meta_element.find("task")
        original_size = task_element.find("original_size")
        self.frame_height = float(original_size.find("height").text)
        self.frame_width = float(original_size.find("width").text)
        print(self.frame_height, self.frame_width)

    def generate_frame_timestamp_map(self):
        self.frame_timestamp_ms_map = self.get_frame_timestamp_ms_map(self.video_uuid)

    def parse(self):
        self.generate_frame_timestamp_map()
        for child in self.root:
            if child.tag == "meta":
                self.parse_meta(child)
            if child.tag == "track":
                self.parse_track(child)

    def dump(self):
        with open(self.output_labels_path, "w") as op_label_file:
            csv_writer = csv.writer(op_label_file)
            for label in list(self.labels):
                csv_writer.writerow(label)


if __name__ == "__main__":
    video_uuid = "e_dock_north_ch22_20201105185845_20201105200042"
    op_labels_path = os.path.join(
        os.environ["BUILD_WORKSPACE_DIRECTORY"],
        "data",
        "auto_ml",
        "pits",
        "batch1",
        "{}.csv".format(video_uuid),
    )
    ip_labels_path = os.path.join(
        os.environ["BUILD_WORKSPACE_DIRECTORY"],
        "data/labels/cvat/annotations/pits/batch1/{}.xml".format(video_uuid),
    )
    converter = AutoMLConveter(video_uuid, ip_labels_path, op_labels_path)
    converter.parse()
    converter.dump()
