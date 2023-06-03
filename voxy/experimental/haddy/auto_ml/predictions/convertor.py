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

from core.structs.actor import Actor
from core.structs.actor import ActorCategory
from core.structs.attributes import Polygon
from core.structs.attributes import RectangleXYXY
from core.structs.frame import Frame
from core.structs.video import Video


class AutoMLPredictionConvertor:
    def __call__(
        self, input_filename_path, output_file_path, frame_width=960, frame_height=480
    ):
        with open(input_filename_path, "rb") as r:
            predictions = json.load(r)
        r.close()
        video = Video("automl_test_cloud_run_annotations")
        all_annotations = predictions["object_annotations"]
        frames_timestamp_mp_ns = {}
        actor_id = 0
        for actor_annotation in all_annotations:
            actor = Actor()
            actor.track_id = actor_id
            actor.category = ActorCategory.PIT
            actor.confidence = actor_annotation["confidence"]
            frames = actor_annotation["frames"]
            # import pdb; pdb.set_trace()
            for frame in frames:
                x_min = 0.0
                y_min = 0.0
                x_max = frame_width
                y_max = frame_height
                if "x_min" in frame["normalized_bounding_box"]:
                    x_min = frame["normalized_bounding_box"]["x_min"]
                if "y_min" in frame["normalized_bounding_box"]:
                    y_min = frame["normalized_bounding_box"]["y_min"]
                if "x_max" in frame["normalized_bounding_box"]:
                    x_max = frame["normalized_bounding_box"]["x_max"]
                if "y_max" in frame["normalized_bounding_box"]:
                    y_max = frame["normalized_bounding_box"]["y_max"]
                vertices = [
                    x_min * frame_width,
                    y_min * frame_height,
                    x_max * frame_width,
                    y_max * frame_height,
                ]
                actor.polygon = RectangleXYXY.from_list(vertices).to_polygon()
                seconds = 0
                nanoseconds = 0
                if "seconds" in frame["time_offset"]:
                    seconds = int(frame["time_offset"]["seconds"])
                if "nanos" in frame["time_offset"]:
                    nanoseconds = int(frame["time_offset"]["nanos"])
                timestamp_ns = seconds * 1000000000 + nanoseconds
                if timestamp_ns in frames_timestamp_mp_ns:
                    frame = frames_timestamp_mp_ns[timestamp_ns]
                else:
                    frame_relative_timestamp_s = float(
                        float(timestamp_ns) / 1000000000.0
                    )
                    frame_relative_timestamp_ms = float(float(timestamp_ns) / 1000000.0)
                    frame_number = len(frames_timestamp_mp_ns.values())
                    frame = Frame(
                        frame_number,
                        frame_relative_timestamp_s,
                        frame_relative_timestamp_ms,
                    )
                    frames_timestamp_mp_ns[timestamp_ns] = frame

                frame.actors.append(actor)
            actor_id = actor_id + 1

        for frame in frames_timestamp_mp_ns.values():
            video.frames.append(frame)

        with open(output_file_path, "w") as w:
            json.dump(video.to_dict(), w)
        w.close()


if __name__ == "__main__":
    predictions_converter = AutoMLPredictionConvertor()
    ip_path = os.path.join(
        os.environ["BUILD_WORKSPACE_DIRECTORY"],
        "data",
        "scratch",
        "haddy",
        "automl",
        "predictions",
        "{}.json".format(
            "prediction-forklift_20201122031207-2020-11-27T20_10_11.608235Z_e_dock_north_ch22_20201104000155_20201104040203_1"
        ),
    )
    op_path = os.path.join(
        os.environ["BUILD_WORKSPACE_DIRECTORY"],
        "data",
        "scratch",
        "haddy",
        "automl",
        "converted_predictions",
        "{}.json".format(
            "prediction-forklift_20201122031207-2020-11-27T20_10_11.608235Z_e_dock_north_ch22_20201104000155_20201104040203_1"
        ),
    )
    predictions_converter(ip_path, op_path)
