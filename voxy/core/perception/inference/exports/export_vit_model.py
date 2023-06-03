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

import numpy as np
import torch
from loguru import logger
from transformers import (
    ViTConfig,
    ViTFeatureExtractor,
    ViTForImageClassification,
)

from core.utils.yaml_jinja import load_yaml_with_jinja
from lib.infra.utils.resolve_model_path import resolve_model_path


def export_vit_model(model_path: str, num_labels: int) -> str:
    """
    Exports the vit model in the given config file to jit format

    Args:
        model_path (str): the absolute path to the model
        num_labels: the number of labels for the model

    Returns:
        str: the jit saved model path
    """
    configuration = ViTConfig(torchscript=True)
    configuration.num_labels = num_labels
    vest_classifier_model = ViTForImageClassification(configuration)
    vest_classifier_model.load_state_dict(torch.load(model_path))
    vest_classifier_model = vest_classifier_model.cuda().eval()
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k", return_tensors="pt"
    )
    logger.info("Loaded model")
    # try to trace the model
    random_actor_crops = [np.random.rand(480, 960, 3) for _ in range(16)]
    trace_inputs = feature_extractor(random_actor_crops, return_tensors="pt")[
        "pixel_values"
    ].cuda()
    logger.info("Traced inputs, exporting through JIT")
    traced_model = torch.jit.trace(vest_classifier_model, trace_inputs)
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
        default="configs/cameras/uscold/quakertown/0001/cha.yaml",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        help=("The number of labels of the model"),
        required=False,
        default=2,
    )
    parser.add_argument(
        "--is_vest",
        action="store_true",
        help="Exports safety vest model",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--is_hat",
        action="store_true",
        help="Exports hard hat model",
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = load_yaml_with_jinja(args.config_path)
    if args.is_vest:
        model_path_ = resolve_model_path(
            config["perception"]["vest_classifier"]["model_path"]
        )
    elif args.is_hat:
        model_path_ = resolve_model_path(
            config["perception"]["hat_classifier"]["model_path"]
        )
    else:
        raise ValueError("You must specify either --is_vest or --is_hat")
    model = export_vit_model(model_path_, args.num_labels)
