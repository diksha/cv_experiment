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

import onnx
import onnxsim
import tensorrt as trt
import torch
import torch.onnx.verification
from loguru import logger

from third_party.vit_pose.configs.ViTPose_base_coco_256x192 import (
    model as model_cfg,
)
from third_party.vit_pose.models.model import ViTPose


def export_trt(
    model: torch.nn.Module, model_base_name: str, build_with_fp16: bool
):
    """
    Exports the model into trt format

    Args:
        model (torch.nn.Module): the model to export (vit pose model)
        model_base_name(str): the base name of the model
        build_with_fp16 (bool): whether to build with fp16

    Raises:
        RuntimeError: if the simplification fails
    """
    # Define the example input shape

    base_input_shape = (3, 256, 192)
    input_shape = (1, *base_input_shape)

    # Export the model to ONNX
    dummy_input = torch.ones(*input_shape).float().cuda() / 255 - 0.5
    onnx_path = f"{model_base_name}.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model.cpu(),
            dummy_input.cpu(),
            onnx_path,
            opset_version=12,
            verbose=False,
            do_constant_folding=True,
            input_names=["input_0"],
            output_names=["output_0"],
            dynamic_axes={
                "input_0": {0: "batch_size"},  # variable length axes
                "output_0": {0: "batch_size"},
            },
        )
    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    onnx.save(model_onnx, onnx_path)
    simplify = False
    if simplify:

        model_onnx, check = onnxsim.simplify(model_onnx)
        if not check:
            raise RuntimeError("Simplify failed")
        onnx.save(model_onnx, onnx_path)

    # Create a TensorRT builder and network
    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt_builder = trt.Builder(trt_logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    trt_network = trt_builder.create_network(explicit_batch)

    # Create a TensorRT parser and parse the ONNX model
    trt_parser = trt.OnnxParser(trt_network, trt_logger)
    with open(onnx_path, "rb") as model_file:
        trt_parser.parse(model_file.read())

    # Set up TensorRT builder configurations
    config = trt_builder.create_builder_config()
    config.max_workspace_size = 4 * 1 << 30

    inputs = [trt_network.get_input(i) for i in range(trt_network.num_inputs)]
    outputs = [
        trt_network.get_output(i) for i in range(trt_network.num_outputs)
    ]
    for inp in inputs:
        logger.info(f' input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        logger.info(f' output "{out.name}" with shape{out.shape} {out.dtype}')

    # Set the dynamic shape profile
    profile = trt_builder.create_optimization_profile()
    for inp in inputs:
        profile.set_shape(
            inp.name,
            min=(
                1,
                *base_input_shape,
            ),
            opt=(
                8,
                *base_input_shape,
            ),
            max=(
                64,
                *base_input_shape,
            ),
        )
    config.add_optimization_profile(profile)

    # Save the TensorRT engine to a file
    engine_path = f"{model_base_name}.engine"

    if trt_builder.platform_has_fast_fp16 and build_with_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    logger.success(f"Generating TRT Engine '{engine_path}'")
    with trt_builder.build_engine(trt_network, config) as engine, open(
        engine_path, "wb"
    ) as out_file:
        out_file.write(engine.serialize())

    logger.success(f"TensorRT engine saved to '{engine_path}'")


def parse_args() -> argparse.Namespace:
    """
    Parses commandline args

    Returns:
        argparse.Namespace: commandline args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help=("Path to the model to export. "),
        required=True,
    )
    # add boolean argument for fp16
    parser.add_argument(
        "--fp16",
        action="store_true",
        help=("Whether to use fp16 for the engine. "),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    base = model_path.split(".", maxsplit=1)[0]
    torch_model = ViTPose(model_cfg)
    torch_model.eval().to("cuda")
    checkpoint = torch.load(  # trunk-ignore(semgrep/trailofbits.python.pickles-in-pytorch.pickles-in-pytorch,pylint/C0301)
        model_path
    )
    torch_model.load_state_dict(checkpoint["state_dict"])
    export_trt(torch_model, base, args.fp16)
