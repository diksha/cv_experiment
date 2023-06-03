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
import json
import os

import cv2

from core.structs.actor import Actor
from core.structs.attributes import RectangleXYWH
from core.structs.frame import Frame
from core.structs.video import Video


class GalleryExtractor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_uuid = str(os.path.basename(args.VIDEO_PATH).replace(".mp4", ""))
        self._frame_ts_label_dict = {}
        self.load_and_store_labels()

    def run(self, args):
        vcap = cv2.VideoCapture(self.video_path)
        frame_id = 0
        track_instance_mp = {}
        while True:
            ret, frame = vcap.read()
            if frame is not None:
                frame_id += 1
                if frame_id % args.frame_interval:
                    continue
                ts_ms = vcap.get(cv2.CAP_PROP_POS_MSEC)
                frame_labels = self.get_label_at_ts_ms(ts_ms)
                if frame_labels is not None:
                    for actor in frame_labels.actors:
                        polygon = RectangleXYWH.from_polygon(actor.polygon)
                        x, y, w, h = [int(i) for i in polygon.to_list()]
                        actor_image = frame[y : y + h, x : x + w]
                        track_id = str(actor.track_id)
                        if track_id in track_instance_mp.keys():
                            track_instance_mp[track_id] += 1
                        else:
                            track_instance_mp[track_id] = 0
                        filename = os.path.join(
                            args.gallery_path,
                            track_id
                            + "_c"
                            + str(args.camera_id)
                            + "_i"
                            + str(track_instance_mp[track_id])
                            + ".jpg",
                        )
                        cv2.imwrite(filename, actor_image)
            else:
                break

    def get_label_at_ts_ms(self, ts_ms):
        for i in self._frame_ts_label_dict.keys():
            # Within 1ms.
            if ts_ms - 1 <= i * 1000 and ts_ms + 1 >= i * 1000:
                return self._frame_ts_label_dict.get(i)

    def load_and_get_labels_from_path(self, path):
        with open(path) as label_file:
            label_dict = json.load(label_file)
            video_labels = Video.from_dict(label_dict)
        return video_labels

    def load_and_store_labels(self):
        label_file_path = os.path.join(
            os.environ["BUILD_WORKSPACE_DIRECTORY"],
            "data",
            "video_labels",
            "{}.json".format(self.video_uuid),
        )
        video_labels = self.load_and_get_labels_from_path(label_file_path)
        self._frame_ts_label_dict = {
            frame.relative_timestamp_s: frame for frame in video_labels.frames
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--frame_interval", type=int, required=False, default=30)
    parser.add_argument("--camera_id", type=int, required=False, default=1)
    parser.add_argument("--gallery_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gallery_extractor = GalleryExtractor(args.VIDEO_PATH)
    gallery_extractor.run(args)
