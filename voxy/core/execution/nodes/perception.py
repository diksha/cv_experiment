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

import os

import numpy
import torch
import yaml
from loguru import logger

from core.execution.nodes.abstract import AbstractNode
from core.execution.utils.graph_config_utils import (
    get_gpu_runtime_from_graph_config,
    get_head_covering_type_from_graph_config,
)
from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.perception.aisle_end.aisle_perception import AislePerception
from core.perception.calibration.world_perspective_model import (
    WorldPerspectiveModel,
)
from core.perception.carry_object.carry_object_classifier import (
    CarryObjectClassifier,
)
from core.perception.detector_tracker.tracker import DetectorTracker
from core.perception.detector_tracker.yolo_detector import YoloDetector
from core.perception.door.state_classifier_vision import DoorStateClassifier
from core.perception.driving_area.driving_area_perception import (
    DrivingAreaPerception,
)
from core.perception.hat.hat_classifier import HatClassifier
from core.perception.intersection.intersection_perception import (
    IntersectionPerception,
)
from core.perception.motion.motion_zone_detector import MotionZoneDetector
from core.perception.no_ped_zone.no_ped_zone_perception import (
    NoPedZonePerception,
)
from core.perception.obstruction.obstruction_segmenter import (
    ObstructionSegmenter,
)
from core.perception.pose.activity_classifier import ActivityClassifier
from core.perception.pose.api import PoseModel
from core.perception.pose.lift_classifier import LiftClassifier
from core.perception.pose.reach_classifier import ReachClassifier
from core.perception.pose.vit_pose import ViTPoseModel
from core.perception.spill.spill_segmenter import SpillSegmenter
from core.perception.vest.vest_classifier import VestClassifier
from core.structs.actor import Actor, ActorCategory
from core.structs.frame import Frame, StampedImage
from core.utils.actionable_region_utils import person_inside_actionable_region
from core.utils.logging.proto_node import proto_node


@proto_node()
class PerceptionNode(AbstractNode):
    USE_VIT_POSE: bool = False

    def __init__(
        self, config: dict, perception_runner_context: PerceptionRunnerContext
    ) -> None:
        self._camera_uuid = config["camera_uuid"]
        self._gpu_runtime = get_gpu_runtime_from_graph_config(config)
        self._triton_server_url = perception_runner_context.triton_server_url
        detector = YoloDetector.from_config(
            config=config, perception_runner_context=perception_runner_context
        )
        self._detector_tracker = DetectorTracker(self._camera_uuid, detector)
        calibration_config = self.get_calibration_config(config)

        self._world_perspective_model = (
            WorldPerspectiveModel(calibration_config)
            if calibration_config is not None
            else None
        )

        if self.USE_VIT_POSE:
            self._pose_model = ViTPoseModel(
                model_path=config["perception"]["pose"]["model_path"],
                gpu_runtime=self._gpu_runtime,
                triton_server_url=self._triton_server_url,
            )
        else:
            self._pose_model = (
                PoseModel(
                    model_path=config["perception"]["pose"]["model_path"],
                    gpu_runtime=self._gpu_runtime,
                    triton_server_url=self._triton_server_url,
                )
                if config["perception"].get("pose", {}).get("enabled", False)
                is True
                else None
            )
        self._carry_object_classifier = (
            CarryObjectClassifier(
                model_path=config["perception"]["carry_object_classifier"][
                    "model_path"
                ],
                prediction2class=config["perception"]
                .get("carry_object_classifier", {})
                .get("prediction2class", {"NOT_CARRYING": 0, "CARRYING": 1}),
                min_actor_pixel_area=config["perception"]
                .get("carry_object_classifier", {})
                .get("min_actor_pixel_area", None),
                gpu_runtime=self._gpu_runtime,
                triton_server_url=self._triton_server_url,
            )
            if config["perception"]
            .get("carry_object_classifier", {})
            .get("enabled", False)
            is True
            else None
        )
        self._vest_classifier = (
            VestClassifier(
                model_path=config["perception"]["vest_classifier"][
                    "model_path"
                ],
                classification_model_type=(
                    config["perception"]["vest_classifier"]["model_type"]
                    if config["perception"]
                    .get("vest_classifier", {})
                    .get("model_type")
                    else None
                ),
                prediction_to_class=config["perception"]
                .get("vest_classifier", {})
                .get("prediction2class", {"NO_VEST": 0, "VEST": 1}),
                min_actor_pixel_area=config["perception"]
                .get("vest_classifier", {})
                .get("min_actor_pixel_area", None),
                gpu_runtime=self._gpu_runtime,
                triton_server_url=self._triton_server_url,
            )
            if config["perception"]
            .get("vest_classifier", {})
            .get("enabled", False)
            is True
            else None
        )
        self._hat_classifier = (
            HatClassifier(
                model_path=config["perception"]["hat_classifier"][
                    "model_path"
                ],
                prediction_to_class=config["perception"]
                .get("vest_classifier", {})
                .get("prediction2class", {"NO_HAT": 0, "HAT": 1}),
                is_classification_by_detection=config["perception"][
                    "hat_classifier"
                ]["is_classification_by_detection"],
                min_actor_pixel_area=config["perception"]
                .get("hat_classifier", {})
                .get("min_actor_pixel_area", None),
                gpu_runtime=self._gpu_runtime,
                head_covering_type=get_head_covering_type_from_graph_config(
                    config
                ),
                triton_server_url=self._triton_server_url,
            )
            if config["perception"]
            .get("hat_classifier", {})
            .get("enabled", False)
            is True
            else None
        )
        self._door_classifier = (
            DoorStateClassifier(
                camera_uuid=self._camera_uuid,
                model_path=config["perception"]["door_classifier"][
                    "model_path"
                ],
                model_type=config["perception"]["door_classifier"].get(
                    "model_type", "resnet"
                ),
                config=config["perception"]["door_classifier"].get(
                    "config", {}
                ),
                gpu_runtime=self._gpu_runtime,
                triton_server_url=self._triton_server_url,
            )
            if config["perception"]
            .get("door_classifier", {})
            .get("enabled", False)
            is True
            else None
        )

        self._intersection_perception = IntersectionPerception(
            camera_uuid=self._camera_uuid,
        )

        self._aisle_perception = AislePerception(
            camera_uuid=self._camera_uuid,
        )

        self._nopedzone_perception = NoPedZonePerception(
            camera_uuid=self._camera_uuid,
        )

        self._drivingarea_perception = DrivingAreaPerception(
            camera_uuid=self._camera_uuid,
        )

        self._lift_classifier = (
            LiftClassifier(
                config["perception"]["lift_classifier"]["model_path"],
                classification_model_type=config["perception"][
                    "lift_classifier"
                ].get("model_type"),
            )
            if config["perception"].get("pose", {}).get("enabled") is True
            and config["perception"].get("lift_classifier", {}).get("enabled")
            is True
            else None
        )

        self._reach_classifier = (
            ReachClassifier(
                config["perception"]["reach_classifier"]["model_path"],
                classification_model_type=(
                    config["perception"]["reach_classifier"]["model_type"]
                    if config["perception"]
                    .get("reach_classifier", {})
                    .get("model_type")
                    else None
                ),
                gpu_runtime=self._gpu_runtime,
                triton_server_url=self._triton_server_url,
            )
            if config["perception"].get("pose", {}).get("enabled") is True
            and config["perception"].get("reach_classifier", {}).get("enabled")
            is True
            else None
        )
        self._activity_classifier = (
            ActivityClassifier(
                config["perception"]["pose_classifier"]["model_path"],
            )
            if config["perception"].get("pose", {}).get("enabled") is True
            and config["perception"].get("pose_classifier", {}).get("enabled")
            is True
            else None
        )
        self._spill_segmenter = (
            SpillSegmenter.from_config(
                config=config,
                perception_runner_context=perception_runner_context,
            )
            if config["perception"].get("spill", {}).get("enabled", False)
            is True
            else None
        )
        self._motion_zone_detector = (
            MotionZoneDetector(
                motion_detector_type=config["perception"][
                    "motion_zone_detection"
                ].get("detector_type", "mog2"),
                camera_uuid=self._camera_uuid,
                config=config["perception"]["motion_zone_detection"],
            )
            if config["perception"]
            .get("motion_zone_detection", {})
            .get("enabled", False)
            is True
            else None
        )
        self._obstruction_segmenter = (
            ObstructionSegmenter.from_config(config)
            if config["perception"]
            .get("obstruction_segmenter", {})
            .get("enabled", False)
            is True
            else None
        )

    def get_calibration_config(self, config: dict) -> dict:
        """Get calibration config

        Args:
            config (dict): the input config file

        Returns:
            dict: calibration config
        """
        calibration_config = None
        # TODO: use a database query to grab this calibration
        candidate_calibration_file = os.path.join(
            "configs/cameras/", config["camera_uuid"] + "_calibration.yaml"
        )
        if os.path.exists(candidate_calibration_file):
            with open(
                candidate_calibration_file,
                "r",
                encoding="utf8",
            ) as calibration_config_file:
                calibration_config = yaml.safe_load(calibration_config_file)
                logger.info("Loaded calibration config")
        else:
            logger.warning(
                f"Could not find calibration file @ {candidate_calibration_file} "
            )
        return calibration_config

    def filter_actors_outside_actionable_region(
        self, actors: list, shape: tuple
    ) -> list:
        """Filter actors that are outside the actionable regions

        Args:
            actors (list): list of actors
            shape (tuple): shape of image

        Returns:
            list: list of actors is in actionable regions
        """

        def is_in_actionable_region(actor: Actor) -> bool:
            """Check if person is in actionable region

            Args:
                actor (Actor): actor of interest

            Returns:
                _bool_: boolean of actor is inside actionable region
            """

            if actor.category != ActorCategory.PERSON:
                return True
            # otherwise we are a person
            return person_inside_actionable_region(
                self._camera_uuid, actor, *shape
            )

        return [actor for actor in actors if is_in_actionable_region(actor)]

    def _add_static_actors(
        self, frame: numpy.ndarray, frame_struct: Frame
    ) -> Frame:
        """
        Adding static actors to the frame struct
        Args:
            frame (numpy.ndarray): input image
            frame_struct (Frame): current frame struct

        Returns:
            Frame: updated frame struct
        """

        if self._intersection_perception is not None:
            frame_struct.actors += self._intersection_perception(frame)

        if self._aisle_perception is not None:
            frame_struct.actors += self._aisle_perception(frame)

        if self._nopedzone_perception is not None:
            frame_struct.actors += self._nopedzone_perception(frame)

        if self._drivingarea_perception is not None:
            frame_struct.actors += self._drivingarea_perception(frame)
        return frame_struct

    def process(self, stamped_image: StampedImage) -> Frame:
        """
        Perception process
        Args:
            stamped_image (StampedImage): the image with timestamp

        Returns:
            Frame: updated frame struct
        """
        frame, frame_epoch_ms = stamped_image.image, stamped_image.timestamp_ms
        detections = self._detector_tracker(frame, frame_epoch_ms)
        detections = self.filter_actors_outside_actionable_region(
            detections, frame.shape[:2]
        )
        frame_struct = Frame(
            frame_number=None,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            relative_timestamp_s=None,
            relative_timestamp_ms=frame_epoch_ms,
            epoch_timestamp_ms=frame_epoch_ms,
        )
        frame_struct.actors = detections
        if self._pose_model is not None:
            frame_struct = self._pose_model(frame, frame_struct)

        if self._lift_classifier is not None:
            frame_struct = self._lift_classifier(frame_struct)

        if self._carry_object_classifier is not None:
            frame_struct = self._carry_object_classifier(frame_struct, frame)

        if self._vest_classifier is not None:
            frame_struct = self._vest_classifier(frame_struct, frame)

        if self._hat_classifier is not None:
            frame_struct = self._hat_classifier(frame_struct, frame)

        if self._door_classifier is not None:
            frame_struct.actors += self._door_classifier(frame)

        if self._reach_classifier is not None:
            frame_struct = self._reach_classifier(frame_struct)

        if self._activity_classifier is not None:
            frame_struct = self._activity_classifier(frame_struct)

        if self._world_perspective_model is not None:
            frame_struct = self._world_perspective_model(frame_struct)

        frame_struct = self._add_static_actors(frame, frame_struct)

        if self._spill_segmenter is not None:
            spill_actors, frame_struct.frame_segments = self._spill_segmenter(
                frame, frame_epoch_ms
            )
            frame_struct.actors.extend(spill_actors)
        if self._motion_zone_detector is not None:
            frame_struct.actors += self._motion_zone_detector(frame)

        if self._obstruction_segmenter is not None:
            obstruction_actors, _ = self._obstruction_segmenter(
                frame, frame_epoch_ms, frame_struct.actors
            )
            frame_struct.actors.extend(obstruction_actors)

        return frame_struct

    def finalize(self):
        """Clear torch gpu cache"""
        torch.cuda.empty_cache()
