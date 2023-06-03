import os

from loguru import logger

from core.labeling.scale.hypothesis_generation.videoplayback_hypothesis.hypothesis_base import (
    DetectorHypothesis,
)
from core.perception.detector_tracker.tracker import DetectorTracker
from core.perception.detector_tracker.yolo_detector import YoloDetector
from core.utils.video_reader import S3VideoReader, S3VideoReaderInput
from core.utils.yaml_jinja import load_yaml_with_jinja
from lib.infra.utils.resolve_model_path import resolve_model_path


class YoloHypothesis(DetectorHypothesis):
    CAMERA_CONFIG_PATH = "configs/cameras"

    def __init__(
        self,
        video_uuid: str,
        camera_uuid: str,
    ):

        super().__init__(video_uuid=video_uuid)

        video_reader_input = S3VideoReaderInput(
            video_path_without_extension=self._video_uuid
        )
        self.video_reader = S3VideoReader(video_reader_input)

        config_path = os.path.join(
            self.CAMERA_CONFIG_PATH, f"{camera_uuid}.yaml"
        )
        camera_config = load_yaml_with_jinja(config_path)
        camera_config["perception"]["detector_tracker"][
            "model_path"
        ] = resolve_model_path(
            camera_config["perception"]["detector_tracker"]["model_path"]
        )
        detector = YoloDetector.from_config(camera_config)

        self._detector_tracker = DetectorTracker(camera_uuid, detector)

    def process(self):
        """Generate scale hypothesis given a detector"""
        logger.info(f"Processing {self._video_uuid}")
        annotations = {}
        frame_count = 1
        # TODO: Confirm that fps lines up with annotation task
        for video_reader_op in self.video_reader.read():
            frame_ms = video_reader_op.relative_timestamp_ms
            frame = video_reader_op.image

            actors = self._detector_tracker(frame, frame_ms)

            for actor in actors:
                frame_hypothesis = self._get_frame_hypothesis(
                    actor, frame_count, frame_ms
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

            frame_count += 1

        logger.info(
            f"Run yolo detection hypothesis on video: {self._video_uuid}"
        )


if __name__ == "__main__":
    hypothesis_generator = YoloHypothesis(
        video_uuid="americold/modesto/0001/cha/20220606_01_doors_0000",
        camera_uuid="americold/modesto/0001/cha",
    )
    hypothesis_generator.process()
