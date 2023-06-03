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
import argparse

import torch
from loguru import logger

from core.perception.carry_object.carry_object_classifier import (
    CarryObjectClassifier,
)
from core.utils.yaml_jinja import load_yaml_with_jinja
from lib.infra.utils.resolve_model_path import resolve_model_path


def export_model(
    config_file: str,
) -> str:
    """
    Exports the pose model in the given config file to jit format

    Args:
        config_file (str): Perception config file to use
    Raises:
       ValueError: if no carry object classifier is found in the config

    Returns:
        str: the jit saved model path
    """
    config = load_yaml_with_jinja(config_file)
    model_path = resolve_model_path(
        config["perception"]["carry_object_classifier"]["model_path"]
    )
    carry_object_classifier = (
        CarryObjectClassifier(
            model_path=model_path,
            prediction2class=config["perception"]
            .get("carry_object_classifier", {})
            .get("prediction2class", {"NOT_CARRYING": 0, "CARRYING": 1}),
            min_actor_pixel_area=config["perception"]
            .get("carry_object_classifier", {})
            .get("min_actor_pixel_area", None),
        )
        if config["perception"]
        .get("carry_object_classifier", {})
        .get("enabled", False)
        is True
        else None
    )
    if carry_object_classifier is None:
        raise ValueError("No carry object classifier found in config")
    logger.info("Loaded model")

    batch_size = 16
    sample_batch = torch.randn(batch_size, 3, 224, 224, device="cuda")

    # # try to trace the model
    traced_model = torch.jit.trace(
        carry_object_classifier.carry_object_model, sample_batch
    )
    jit_path = f"{model_path}.jit"
    traced_model.save(jit_path)
    logger.info(f"Traced output to: {jit_path}")
    return jit_path


def parse_args() -> argparse.Namespace:
    """
    Parses commandline args

    Returns:
        argparse.Namespace: commandline args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help=(
            "The configuration run file. "
            "Defaults to configs/cameras/americold/modesto/0001/cha.yaml."
        ),
        required=False,
        default="configs/cameras/americold/modesto/0001/cha.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_model(args.config_path)
