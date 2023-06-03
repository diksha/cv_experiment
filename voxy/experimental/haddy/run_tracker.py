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
from core.structs.attributes import RectangleXYXY
from core.structs.frame import Frame
from core.structs.video import Video
from core.tracker.structs.tracker_types import TrackerType
from core.tracker.tracker_factory import tracker_factory
from core.utils.drawing.draw_bounding_boxes import draw_bounding_boxes_with_ids


class VideoTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.vdo = cv2.VideoCapture()
        self.detector = detector_factory.create_detector(
            DetectorType.ALPHAPOSE, display=False
        )
        # self.detector = detector_factory.create_detector(DetectorType.YOLOv3, use_cuda=True, display=False)
        self.tracker = tracker_factory.create_tracker(
            TrackerType.DEEPSORT, use_cuda=True, display=False
        )
        self.video_uuid = str(os.path.basename(video_path).replace(".mp4", ""))
        self.video = Video(self.video_uuid)

    def setup_ground_truth_detector(self):
        labels_file = os.path.join(
            os.environ["BUILD_WORKSPACE_DIRECTORY"],
            "data/video_labels/",
            self.video_uuid + ".json",
        )
        fps = self.vdo.get(cv2.CAP_PROP_FPS)
        self.detector = detector_factory.create_detector(
            DetectorType.GROUND_TRUTH,
            labels_file=labels_file,
            display=False,
            video_fps=fps,
        )

    def _write_labels(self):
        data = self.video.to_dict()
        # TODO(haddy): Save runs in different folders based on which detector
        # and tracker are run.
        save_file_path = os.path.join(
            os.environ["BUILD_WORKSPACE_DIRECTORY"],
            "data/scratch/haddy/alphapose_deepsort_truncated",
            self.video.uuid + ".json",
        )
        with open(save_file_path, "w") as save_file:
            json.dump(data, save_file, indent=4, sort_keys=True)
        print("Wrote out labels to ", save_file_path)

    def _get_bbox_xyxy_and_ids_from_tracker_op(self, actors):
        bboxes_xyxy = []
        ids = []
        for i, actor in enumerate(actors):
            if actor.polygon and actor.track_id:
                bboxes_xyxy.append(RectangleXYXY.from_polygon(actor.polygon).to_list())
                ids.append(actor.track_id)
        return bboxes_xyxy, ids

    def run(self):
        self.vdo.open(self.video_path)
        # TODO(haddy): Change the name based on the detector and tracker
        # running
        self.video_writer = cv2.VideoWriter(
            "alphapose_deepsort.mp4",
            0x7634706D,
            10,
            (int(self.vdo.get(3)), int(self.vdo.get(4))),
        )
        # override detector with ground truth detector. Note call only after calling
        # self.video.open(video_path)
        # self.setup_ground_truth_detector()
        while self.vdo.grab():
            ts_ms = self.vdo.get(cv2.CAP_PROP_POS_MSEC)
            ts_ms = float(ts_ms / 1000.0)
            frame = Frame(relative_timestamp_s=ts_ms)
            _, img = self.vdo.retrieve()
            detector_actors = self.detector(img)
            if len(detector_actors) > 0:
                tracker_actors = self.tracker.update(detector_actors, img)
                frame.actors = tracker_actors
                bboxes_xyxy, ids = self._get_bbox_xyxy_and_ids_from_tracker_op(
                    tracker_actors
                )
                image = draw_bounding_boxes_with_ids(img, bboxes_xyxy, ids)
            self.video.frames.append(frame)
            self.video_writer.write(image)
        self.vdo.release()
        self.video_writer.release()
        # self._write_labels()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_tracker = VideoTracker(args.VIDEO_PATH)
    video_tracker.run()
