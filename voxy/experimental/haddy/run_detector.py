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

from core.detector.detector_factory import detector_factory
from core.detector.structs.detector_types import DetectorType
from core.structs.actor import Actor
from core.structs.frame import Frame
from core.structs.video import Video


class VideoDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.vdo = cv2.VideoCapture()
        # self.detector = detector_factory.create_detector(DetectorType.YOLOv3, display=True, use_cuda=True)
        self.detector = detector_factory.create_detector(
            DetectorType.ALPHAPOSE, display=True
        )
        self.video_uuid = str(os.path.basename(video_path).replace(".mp4", ""))
        self.video = Video(self.video_uuid)

    def _write_labels(self):
        data = self.video.to_dict()
        save_file_path = os.path.join(
            os.environ["BUILD_WORKSPACE_DIRECTORY"],
            "data/scratch/haddy",
            self.video.uuid + ".json",
        )
        with open(save_file_path, "w") as save_file:
            json.dump(data, save_file, indent=4, sort_keys=True)
        print("Wrote out labels to ", save_file_path)

    def run(self):
        self.vdo.open(self.video_path)
        while self.vdo.grab():
            ts_ms = self.vdo.get(cv2.CAP_PROP_POS_MSEC)
            ts_ms = float(ts_ms / 1000.0)
            frame = Frame(relative_timestamp_s=ts_ms)
            _, img = self.vdo.retrieve()
            det_ops = self.detector(img)
            frame.actors = det_ops
            self.video.frames.append(frame)
        self._write_labels()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_detector = VideoDetector(args.VIDEO_PATH)
    video_detector.run()
