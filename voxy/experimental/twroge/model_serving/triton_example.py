#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import torch
from sortedcontainers import SortedDict

from core.structs.actor import ActorCategory
from core.utils.yaml_jinja import load_yaml_with_jinja
from lib.infra.utils.resolve_model_path import resolve_model_path
from lib.ml.inference.tasks.door_state.vanilla_resnet.triton import (
    TritonInferenceProvider as DoorTritonInferenceProvider,
)
from lib.ml.inference.tasks.object_detection_2d.yolov5.triton import (
    TritonInferenceProvider as YOLOTritonInferenceProvider,
)

# model protos are defined here:
# https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto


def test_yolo():
    """
    Tests yolo on the triton remote inference server
    """
    camera_uuid = "americold/modesto/0011/cha"
    config_file = f"configs/cameras/{camera_uuid}.yaml"
    config = load_yaml_with_jinja(config_file)
    model_path = config["perception"]["detector_tracker"]["model_path"]
    height = config["perception"]["detector_tracker"]["height"]
    width = config["perception"]["detector_tracker"]["width"]
    input_shape = (height, width)
    actor_to_class = config["perception"]["detector_tracker"]["actor2class"]
    classes = SortedDict(
        {
            class_id: getattr(ActorCategory, category)
            for category, class_id in actor_to_class.items()
        }
    )
    inference_provider = YOLOTritonInferenceProvider(
        model_path=resolve_model_path(model_path),
        input_shape=input_shape,
        classes=classes,
        device=torch.device("cpu"),
    )
    for _ in range(100):
        example_input = torch.randint(
            low=0, high=255, size=(1, height, width, 3), dtype=torch.uint8
        )
        inference_provider.process(example_input)
    print("Ran inference on 100 samples for YOLO")


def test_door():
    """
    Basic program to load a torch model into
    triton and run inference on it
    """
    camera_uuid = "americold/modesto/0011/cha"
    config_file = f"configs/cameras/{camera_uuid}.yaml"
    config = load_yaml_with_jinja(config_file)
    model_path = config["perception"]["door_classifier"]["model_path"]
    inference_provider = DoorTritonInferenceProvider(
        model_path=resolve_model_path(model_path),
        camera_uuid=camera_uuid,
        config={
            "state2class": {
                "closed": 0,
                "open": 1,
                "partially_open": 2,
            },
            "runtime_preprocessing_transforms": {
                "bgr2rgb": True,
            },
            "postprocessing_transforms": {
                "padding": 30,
            },
        },
        device=torch.device("cpu"),
    )
    for _ in range(100):
        example_input = torch.randint(
            low=0, high=255, size=(4, 480, 960, 3), dtype=torch.uint8
        )
        inference_provider.process(example_input)
    print("Ran inference on 100 samples for Door")


if __name__ == "__main__":
    test_door()
    test_yolo()
