import os
import shutil

from google.protobuf.json_format import ParseDict
from google.protobuf.text_format import MessageToString
from loguru import logger
from tritonclient.grpc import model_config_pb2

from lib.infra.utils.resolve_model_path import resolve_model_path
from lib.ml.inference.backends.triton import PLATFORM_TARGET_NAME_MAP
from lib.ml.inference.tasks.object_detection_2d.yolov5.post_processing_model import (
    transform_and_post_process,
)
from lib.ml.inference.tasks.object_detection_2d.yolov5.pre_processing_model import (
    preprocess_image,
)
from lib.ml.inference.tasks.object_detection_2d.yolov5.utils import (
    get_inference_output_shape,
)

CONFIG_FILE_NAME = "config.pbtxt"


def generate_preprocessing_config(model_repository_path: str, new_shape: list):
    """
    Generates the preprocessing config for the yolov5 model
    in the model repo

    Args:
        model_repository_path (str): the path to the model repo
        new_shape (list): the new shape that the preprocessing model is converting to

    Returns:
        str: the preprocessing model name
    """
    yolov5_preprocessing_config = {
        "name": "yolov5_preprocessing",
        "platform": "pytorch_libtorch",
        "max_batch_size": 16,
        "input": [
            {
                "name": "INPUT_0",
                "data_type": "TYPE_UINT8",
                "format": "FORMAT_NCHW",
                "dims": [3, -1, -1],
            },
            {
                "name": "INPUT_1",
                "data_type": "TYPE_INT32",
                "dims": [2],
            },
        ],
        "output": [
            {
                "name": "OUTPUT_0",
                "data_type": "TYPE_FP16",
                "dims": [3, *new_shape],
            },
            {
                "name": "OUTPUT_1",
                "data_type": "TYPE_FP32",
                "dims": [2],
            },
            {
                "name": "OUTPUT_2",
                "data_type": "TYPE_FP32",
                "dims": [2],
            },
        ],
    }
    yolov5_preprocessing_config_pb = model_config_pb2.ModelConfig()
    ParseDict(yolov5_preprocessing_config, yolov5_preprocessing_config_pb)

    # Save preprocessing to model repo
    preprocessing_model_name = yolov5_preprocessing_config_pb.name
    repo_relative_path = os.path.join(
        model_repository_path, preprocessing_model_name
    )
    if os.path.exists(repo_relative_path):
        shutil.rmtree(repo_relative_path)
    repo_model_path = os.path.join(repo_relative_path, "1")
    os.makedirs(repo_model_path)
    model_path = os.path.join(
        repo_model_path,
        PLATFORM_TARGET_NAME_MAP[yolov5_preprocessing_config_pb.platform],
    )
    preprocess_image.save(model_path)
    with open(
        os.path.join(repo_relative_path, CONFIG_FILE_NAME),
        "w",
        encoding="utf-8",
    ) as out_file:
        out_file.write(
            MessageToString(
                yolov5_preprocessing_config_pb,
                use_short_repeated_primitives=True,
            )
        )
    return preprocessing_model_name


def setup_yolo_model_config(
    model_repository_path: str,
    relative_model_path: str,
    input_shape: list,
    n_classes: int,
) -> str:
    """
    Generates the yolo model in the model repo

    Args:
        model_repository_path (str): the path to the model repo
        relative_model_path (str): the relative path to the model
        input_shape (list): the input shape of the model
        n_classes (int): the number of classes in the model

    Returns:
        (str): the yolo model name
    """
    # Input Shape and Classes for Modesto Model
    n_anchors, n_anchor_points = get_inference_output_shape(
        input_shape, n_classes
    )
    yolov5_config = {
        "name": "americold_modesto_yolov5",
        "platform": "tensorrt_plan",
        "max_batch_size": 16,
        "input": [
            {
                "name": "images",
                "data_type": "TYPE_FP16",
                "format": "FORMAT_NCHW",
                "dims": [3, *input_shape],
            }
        ],
        "output": [
            {
                "name": "output0",
                "data_type": "TYPE_FP16",
                "dims": [n_anchors, n_anchor_points],
            }
        ],
    }
    yolov5_config_pb = model_config_pb2.ModelConfig()
    ParseDict(yolov5_config, yolov5_config_pb)
    yolo_model_name = yolov5_config_pb.name
    repo_relative_path = os.path.join(model_repository_path, yolo_model_name)
    if os.path.exists(repo_relative_path):
        shutil.rmtree(repo_relative_path)
    repo_model_path = os.path.join(repo_relative_path, "1")
    os.makedirs(repo_model_path)
    model_path = os.path.join(
        repo_model_path, PLATFORM_TARGET_NAME_MAP[yolov5_config_pb.platform]
    )
    shutil.copyfile(relative_model_path, model_path)
    with open(
        os.path.join(repo_relative_path, CONFIG_FILE_NAME),
        "w",
        encoding="utf-8",
    ) as _out_file:
        _out_file.write(
            MessageToString(
                yolov5_config_pb, use_short_repeated_primitives=True
            )
        )
    return yolo_model_name, n_anchors, n_anchor_points


def setup_postprocessing_for_triton(
    model_repository_path: str, n_anchors: int, n_anchor_points: int
) -> str:
    """
    Generates the postprocessing model in the model repo

    Args:
        model_repository_path (str): the path to the model repo
        n_anchors (int): the number of anchors in the model
        n_anchor_points (int): the number of anchor points in the model

    Returns:
        str: the name of the postprocessing model
    """
    yolov5_postprocessing_config = {
        "name": "yolov5_postprocessing",
        "platform": "pytorch_libtorch",
        "max_batch_size": 16,
        "input": [
            {
                "name": "input0",  # prediction
                "data_type": "TYPE_FP16",
                "dims": [n_anchors, n_anchor_points],
            },
            {
                "name": "input1",  # offset
                "data_type": "TYPE_FP32",
                "dims": [2],
            },
            {
                "name": "input2",  # scale
                "data_type": "TYPE_FP32",
                "dims": [2],
            },
            {
                "name": "input3",  # classes
                "data_type": "TYPE_INT32",
                "dims": [2],
            },
            {
                "name": "input4",  # conf
                "data_type": "TYPE_FP16",
                "dims": [1],
            },
            {
                "name": "input5",  # nms
                "data_type": "TYPE_FP16",
                "dims": [1],
            },
        ],
        "output": [
            {
                "name": "output0",  # num obsercvations
                "data_type": "TYPE_INT64",
                "dims": [1],
            },
            {
                "name": "output1",  # observations
                "data_type": "TYPE_FP32",
                "dims": [9],
            },
        ],
    }

    yolov5_postprocessing_config_pb = model_config_pb2.ModelConfig()
    ParseDict(yolov5_postprocessing_config, yolov5_postprocessing_config_pb)
    # Save postprocessing to model repo
    postprocessing_model_name = yolov5_postprocessing_config_pb.name
    repo_relative_path = os.path.join(
        model_repository_path, postprocessing_model_name
    )
    if os.path.exists(repo_relative_path):
        shutil.rmtree(repo_relative_path)
    repo_model_path = os.path.join(repo_relative_path, "1")
    os.makedirs(repo_model_path)
    model_path = os.path.join(
        repo_model_path,
        PLATFORM_TARGET_NAME_MAP[yolov5_postprocessing_config_pb.platform],
    )
    transform_and_post_process.save(model_path)
    with open(
        os.path.join(repo_relative_path, CONFIG_FILE_NAME),
        "w",
        encoding="utf-8",
    ) as _out_file:
        _out_file.write(
            MessageToString(
                yolov5_postprocessing_config_pb,
                use_short_repeated_primitives=True,
            )
        )
    return postprocessing_model_name


def setup_ensemble_triton_repo(
    model_repository_path: str,
    preprocessing_model_name: str,
    yolo_model_name: str,
    postprocessing_model_name: str,
) -> str:
    """
    Sets up the ensemble model in the model repo

    Args:
        model_repository_path (str): the model repo path
        preprocessing_model_name (str): the preprocessing model name
        yolo_model_name (str): the yolo model name
        postprocessing_model_name (str): the postprocessing model name

    Returns:
        str: the name of the ensemble model
    """

    ensemble_config = {
        "name": "yolov5_ensemble",
        "platform": "ensemble",
        "max_batch_size": 16,
        "input": [
            {
                "name": "INPUT_0",  # Image
                "data_type": "TYPE_UINT8",
                "format": "FORMAT_NCHW",
                "dims": [3, -1, -1],
            },
            {
                "name": "INPUT_1",  # New Shape
                "data_type": "TYPE_INT32",
                "dims": [2],
            },
            {
                "name": "INPUT_2",  # Classes
                "data_type": "TYPE_INT32",
                "dims": [2],
            },
            {
                "name": "INPUT_3",  # Confidence
                "data_type": "TYPE_FP16",
                "dims": [1],
            },
            {
                "name": "INPUT_4",  # NMS Threshold
                "data_type": "TYPE_FP16",
                "dims": [1],
            },
        ],
        "output": [
            {
                "name": "output0",  # num obsercvations
                "data_type": "TYPE_INT64",
                "dims": [1],
            },
            {
                "name": "output1",  # observations
                "data_type": "TYPE_FP32",
                "dims": [9],
            },
        ],
        "ensemble_scheduling": {
            "step": [
                {
                    "model_name": preprocessing_model_name,
                    "model_version": 1,
                    "input_map": {
                        "INPUT_0": "INPUT_0",
                        "INPUT_1": "INPUT_1",
                    },
                    "output_map": {
                        "OUTPUT_0": "images",
                        "OUTPUT_1": "offset",
                        "OUTPUT_2": "scale",
                    },
                },
                {
                    "model_name": yolo_model_name,
                    "model_version": 1,
                    "input_map": {
                        "images": "images",
                    },
                    "output_map": {
                        "output0": "predictions",
                    },
                },
                {
                    "model_name": postprocessing_model_name,
                    "model_version": 1,
                    "input_map": {
                        "input0": "predictions",
                        "input1": "offset",
                        "input2": "scale",
                        "input3": "INPUT_2",
                        "input4": "INPUT_3",
                        "input5": "INPUT_4",
                    },
                    "output_map": {
                        "output0": "output0",
                        "output1": "output1",
                    },
                },
            ],
        },
    }
    ensemble_config_pb = model_config_pb2.ModelConfig()
    ParseDict(ensemble_config, ensemble_config_pb)
    ensemble_model_name = ensemble_config_pb.name
    repo_relative_path = os.path.join(
        model_repository_path, ensemble_model_name
    )
    if os.path.exists(repo_relative_path):
        shutil.rmtree(repo_relative_path)
    repo_model_path = os.path.join(repo_relative_path, "1")
    os.makedirs(repo_model_path)
    with open(
        os.path.join(repo_relative_path, CONFIG_FILE_NAME),
        "w",
        encoding="utf-8",
    ) as _out_file:
        _out_file.write(
            MessageToString(
                ensemble_config_pb, use_short_repeated_primitives=True
            )
        )
    return ensemble_model_name


def setup_model(
    model_path: str = "artifacts_2021-12-06-00-00-00-0000-yolo/best_480_960.engine",
    model_repository_path: str = "/tmp/modelrepos/ensemble",  # trunk-ignore(bandit/B108)
    new_shape: tuple = (480, 960),
    n_classes: int = 2,
):
    """
    Sets up all pre/post processing and model configs
    for triton inference server and adds the ensemble model.

    Args:
        model_path (str, optional): yolo model path. Defaults
                   to "artifacts_2021-12-06-00-00-00-0000-yolo/best_480_960.engine".
        model_repository_path (str, optional): the path to
                   the model repo. Defaults to "/tmp/modelrepos/ensemble".
        new_shape (tuple, optional): the new shape of the model. Defaults to (480, 960).
        n_classes (int, optional): the number of classes. Defaults to 2.
    """
    relative_model_path: str = resolve_model_path(model_path)
    # hardcoded for modesto
    input_shape: list = (480, 960)
    new_shape = (480, 960)
    n_classes: int = 2
    preprocessing_model_name = generate_preprocessing_config(
        model_repository_path, new_shape
    )
    yolo_model_name, n_anchors, n_anchor_points = setup_yolo_model_config(
        model_repository_path, relative_model_path, input_shape, n_classes
    )
    post_processing_model_name = setup_postprocessing_for_triton(
        model_repository_path, n_anchors, n_anchor_points
    )
    ensemble_model_name = setup_ensemble_triton_repo(
        model_repository_path,
        preprocessing_model_name,
        yolo_model_name,
        post_processing_model_name,
    )
    logger.success(f"Generated ensemble model {ensemble_model_name}")


if __name__ == "__main__":
    setup_model()
