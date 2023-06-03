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

from core.perception.pose.reach_classifier import ReachClassifier
from core.utils.yaml_jinja import load_yaml_with_jinja
from lib.infra.utils.resolve_model_path import resolve_model_path


def export_model(
    config_file: str,
) -> str:
    """
    Exports the reach classifier model in the given config file to jit format

    Args:
        config_file (str): Perception config file to use

    Returns:
        str: the jit saved model path
    """
    config = load_yaml_with_jinja(config_file)
    model_path = resolve_model_path(
        config["perception"]["reach_classifier"]["model_path"]
    )
    model = ReachClassifier(
        classifier_path=model_path, classification_model_type="DL"
    )
    logger.info("Loaded model")
    # try to trace the model
    sample_batch_size = 128
    trace_inputs = torch.zeros(sample_batch_size, 30, device="cuda").float()
    logger.info("Traced inputs, exporting through JIT")
    traced_model = torch.jit.trace(
        model._pose_classifier, trace_inputs  # trunk-ignore(pylint/W0212)
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
            "Defaults to configs/cameras/americold/modesto/0011/cha.yaml."
        ),
        required=False,
        default="configs/cameras/americold/modesto/0011/cha.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_model(args.config_path)
