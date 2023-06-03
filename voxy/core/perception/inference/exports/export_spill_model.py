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

import cv2
import numpy as np
import torch
from loguru import logger

from core.perception.spill.spill_segmenter import SpillSegmenter
from core.utils.yaml_jinja import load_yaml_with_jinja
from lib.infra.utils.resolve_model_path import resolve_all_model_paths


def export_model(
    config_file: str,
) -> str:
    """
    Exports the spill model in the given config file to jit format

    Args:
        config_file (str): Perception config file to use

    Returns:
        str: the jit saved model path
    """
    config = load_yaml_with_jinja(config_file)
    resolve_all_model_paths(config)
    # instantiate a spill segmenter model
    segmenter = SpillSegmenter.from_config(config)
    random_input_image = np.random.rand(1000, 256, 3)
    model_path = config["perception"]["spill"]["model_path"]
    norm_image = cv2.normalize(
        random_input_image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    tensor_input = segmenter.process_input(norm_image)
    print(type(tensor_input))
    logger.info(
        f"Exporting spill model to jit format with input size:\n {tensor_input.size()}"
    )
    # trace the spill segmenter model and save it
    jit_path = f"{model_path}.jit"
    # Ignore trunk to access private model
    # trunk-ignore-begin(pylint/W0212)
    torch.jit.save(
        torch.jit.trace(segmenter._model, tensor_input),
        jit_path,
    )
    sample_output = segmenter._model(tensor_input)
    # trunk-ignore-end(pylint/W0212)
    logger.info(
        f"Saved jit model to {jit_path} with output size:\n {sample_output.size()}"
    )
    # ensure that the model works with a different aspect ratio
    random_input_image_other_aspect_ratio = np.random.rand(256, 1000, 3)
    norm_image = cv2.normalize(
        random_input_image_other_aspect_ratio,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    aspect_tensor = segmenter.process_input(norm_image)
    jit_model = torch.jit.load(jit_path).eval()
    model_output = jit_model(aspect_tensor)
    if model_output.size()[-2:] == (aspect_tensor.size()[-2:]):
        logger.success("The exported model works with different aspect ratios")
    else:
        logger.error(
            "The exported model does not work with different aspect ratios. "
            "Please check the model and try again"
        )

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
            "Defaults to configs/cameras/americold/modesto/0011/cha.yaml."
        ),
        required=False,
        default="configs/cameras/americold/modesto/0011/cha.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_model(args.config_path)
