import json
import os
import tempfile

import numpy as np
from loguru import logger

from core.labeling.scale.hypothesis_generation.videoplayback_hypothesis.hypothesis_base import (
    DetectorHypothesis,
)
from core.structs.actor import Actor
from core.utils.aws_utils import read_from_s3, upload_file


class AnnotationHypothesis(DetectorHypothesis):
    VERSION = "v1"
    VIDEO_BUCKET = "voxel-consumable-labels"

    def __init__(self, video_uuid: str, is_test=False):
        self.is_test = is_test
        super().__init__(video_uuid=video_uuid)

    def process(self) -> (str, int):
        """Generate scale hypothesis and frame rate for labeling
        given annotation and upload to s3

        Returns:
            (str, int): s3 path to hypothesis, and frame_rate video
            was labeled with

        Raises:
            RuntimeError: If keys are not equally spaced
        """
        logger.info(f"Processing {self._video_uuid}")
        # TODO(diksha): Allow only for cvat tasks.
        # For scale use the functionality to resend with new labels.
        hypothesis_video_struct = json.loads(
            read_from_s3(
                path=os.path.join(
                    "s3://",
                    self.VIDEO_BUCKET,
                    self.VERSION,
                    f"{self._video_uuid}.json",
                )
            )
        )
        frames = hypothesis_video_struct["frames"]
        annotations = {}
        for frame in frames:
            for actor in frame["actors"]:
                if actor["category"] == "PIT" or actor["category"] == "PERSON":
                    continue
                actor = Actor.from_dict(actor)
                frame_hypothesis = self._get_frame_hypothesis(
                    actor,
                    frame["frame_number"],
                    frame["relative_timestamp_ms"],
                )

                if actor.track_uuid not in annotations:
                    annotations[actor.track_uuid] = {
                        "label": actor.category.name,
                        "geometry": "box",
                        "frames": [frame_hypothesis],
                    }
                else:
                    annotations[actor.track_uuid]["frames"].append(
                        frame_hypothesis
                    )
        keys = set()
        for _, value in annotations.items():
            keys.update({frame["key"] for frame in value["frames"]})
        if len(keys) == 1:
            frame_rate = list(keys)[0]
        elif len(set(np.diff(sorted(keys)))) == 1:
            frame_rate = list(set(np.diff(sorted(keys))))[0].item()
        else:
            raise RuntimeError(
                f"Keys are not equally spaced for {self._video_uuid}"
            )

        with tempfile.NamedTemporaryFile() as tmp:
            with open(tmp.name, "w", encoding="utf-8") as outfile:
                json.dump(annotations, outfile)
            s3_path = (
                f"hypothesis/test/{self._video_uuid}.json"
                if self.is_test
                else f"hypothesis/{self._video_uuid}.json"
            )
            bucket = "voxel-datasets"
            upload_file(bucket, tmp.name, s3_path)
            full_s3_path = f"s3://{bucket}/{s3_path}"
            return full_s3_path, frame_rate, annotations
