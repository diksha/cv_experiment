import argparse
import typing
from pathlib import Path

import onnx
import onnxsim
import tensorrt as trt
import torch
from loguru import logger

K_REQUIRED_OPSET = 12


def parse_args() -> typing.Any:
    """Parse Args
    Returns:
        typing.Any: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="path to torchscript weights",
    )
    parser.add_argument(
        "--imgsz", nargs="+", type=int, required=True, help="image (h, w)"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--simplify", action="store_true", help="ONNX: simplify model"
    )
    parser.add_argument(
        "--workspace", type=int, default=4, help="TensorRT workspace size (GB)"
    )
    parser.add_argument(
        "--half", action="store_true", help="FP16 half-precision export"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="verbose logging"
    )
    return parser.parse_args()


def export_onnx(
    ts_model: typing.Any,
    example_input: torch.tensor,
    ts_model_path: Path,
    simplify: bool,
    verbose: bool,
) -> str:
    """Export onnx
    Args:
        ts_model (typing.Any): torchscript model
        example_input (torch.tensor): input tensor
        ts_model_path (Path): torchscript model path
        simplify (bool): onnx simplification
        verbose (bool): verbose logging
    Returns:
        str: onnx model path
    Raises:
        AssertionError: failed to optimize onnx model
    """
    onnx_model_path = ts_model_path.with_suffix(".onnx")
    input_names = ["images"]
    output_names = ["output0"]
    dynamic = {
        input_names[0]: {0: "batch", 2: "height", 3: "width"},
        output_names[0]: {0: "batch", 1: "class"},
    }
    with torch.no_grad():
        example_output = ts_model(example_input)
        torch.onnx.export(
            ts_model,
            example_input,
            onnx_model_path,
            verbose=verbose,
            opset_version=K_REQUIRED_OPSET,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            example_outputs=example_output,
            dynamic_axes=dynamic,
        )
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, onnx_model_path)

    if simplify:
        onnx_model, check = onnxsim.simplify(onnx_model)
        if not check:
            raise AssertionError("onnx optimization failed")
        onnx.save(onnx_model, onnx_model_path)

    return onnx_model_path


def export_engine(
    ts_model: typing.Any,
    example_input: torch.tensor,
    ts_model_path: str,
    dtype: torch.dtype,
    workspace: int,
    simplify: bool,
    verbose: bool,
) -> None:
    """Export TRT
    Args:
        ts_model (typing.Any): torchscript model
        example_input (torch.tensor): input tensor
        ts_model_path (Path): torchscript model path
        dtype (torch.dtype): dtype
        workspace (int): trt gpu memory alloc
        simplify (bool): onnx simplification
        verbose (bool): verbose logging
    Raises:
        RuntimeError: failed to export onnx model
    """
    onnx_model_path = export_onnx(
        ts_model,
        example_input,
        ts_model_path,
        simplify,
        verbose,
    )
    engine_model_path = ts_model_path.with_suffix(".engine")
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_model_path)):
        raise RuntimeError(f"failed to load ONNX file: {onnx_model_path}")
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        logger.info(
            f'TensorRT input "{inp.name}" with shape{inp.shape} {inp.dtype}'
        )
    for out in outputs:
        logger.info(
            f'TensorRT output "{out.name}" with shape{out.shape} {out.dtype}'
        )
    if example_input.shape[0] <= 1:
        logger.warning(
            "TensorRT WARNING dynamic model requires maximum --batch-size argument"
        )
    profile = builder.create_optimization_profile()
    for inp in inputs:
        profile.set_shape(
            inp.name,
            (1, *example_input.shape[1:]),
            (max(1, example_input.shape[0] // 2), *example_input.shape[1:]),
            example_input.shape,
        )
    config.add_optimization_profile(profile)
    if builder.platform_has_fast_fp16 and dtype == torch.float16:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(
        engine_model_path, "wb"
    ) as engine_file:
        engine_file.write(engine.serialize())


def export(
    weights: str,
    imgsz: list,
    batch_size: int = 1,
    simplify: bool = False,
    workspace: int = 4,
    half: bool = False,
    verbose: bool = False,
) -> None:
    """Export TRT
    Args:
        weights (str): torchscript model path
        imgsz (list): image size (h, w)
        batch_size (int): max batch size
        simplify (bool): onnx optimization
        workspace (int): trt gpu memory alloc
        half (bool): use float16
        verbose (bool): verbose logging
    Raises:
        RuntimeError: no gpu
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Exporting TRT models requires GPU")

    imgsz *= 2 if len(imgsz) == 1 else 1
    device = torch.device("cuda:0")
    dtype = torch.float16 if half else torch.float32
    ts_file = Path(weights)
    ts_model = torch.jit.load(str(ts_file)).to(device).to(dtype)
    example_input = (
        torch.randn(batch_size, 3, imgsz[0], imgsz[1], requires_grad=False)
        .to(device)
        .to(dtype)
    )
    export_engine(
        ts_model, example_input, ts_file, dtype, workspace, simplify, verbose
    )


if __name__ == "__main__":
    args = parse_args()
    export(**vars(args))
