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

from core.execution.nodes.abstract import AbstractNode
from core.labeling.label_store.label_reader import LabelReader
from core.structs.video import Video
from core.utils.aws_utils import (
    does_s3_blob_exists,
    read_from_s3,
    upload_fileobj_to_s3,
)


class AnnotationNode(AbstractNode):
    def __init__(self, config):
        self.video_uuid = config["video_uuid"]
        self.cache_key = config["cache_key"]

        self.aws_cache_path = (
            f"s3://voxel-temp/annotations_cache/"
            f"{self.cache_key}/{self.video_uuid}.json"
        )
        # Only update if the cache doesn't exists, never overwrite it.
        self.already_cached = does_s3_blob_exists(self.aws_cache_path)

        gt_labels = LabelReader().read(self.video_uuid)
        self.gt_video_struct = (
            Video.from_dict(json.loads(gt_labels))
            if gt_labels is not None
            else Video(uuid=self.video_uuid)
        )
        self.gt_annotations_map = {
            int(frame.relative_timestamp_ms): frame
            for frame in self.gt_video_struct.frames
        }

        if self.already_cached:
            self.pred_video_struct = Video.from_dict(
                json.loads(read_from_s3(self.aws_cache_path))
            )
        else:
            self.pred_video_struct = Video(uuid=self.video_uuid)

        self.pred_annotations_map = {
            int(frame.relative_timestamp_ms): frame
            for frame in self.pred_video_struct.frames
        }

    def get_pred_annotation(self, relative_timestamp_ms):
        return self.pred_annotations_map.get(int(relative_timestamp_ms))

    def get_gt_annotation(self, relative_timestamp_ms):
        return self.gt_annotations_map.get(int(relative_timestamp_ms))

    def cache_pred_annotation(self, frame_struct):
        self.pred_video_struct.frames.append(frame_struct)
        self.pred_annotations_map[
            int(frame_struct.relative_timestamp_ms)
        ] = frame_struct

    def is_cached(self):
        return self.already_cached

    def finalize(self):
        if self.pred_annotations_map and not self.already_cached:
            upload_fileobj_to_s3(
                self.aws_cache_path,
                json.dumps(self.pred_video_struct.to_dict()).encode("utf-8"),
            )

    def get_pred_annotation_map(self):
        return self.pred_annotations_map
