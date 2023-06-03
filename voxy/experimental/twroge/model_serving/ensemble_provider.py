import numpy as np
import torch
import tritonclient
from loguru import logger
from tritonclient.grpc import InferInput, InferRequestedOutput

from lib.ml.inference.tasks.object_detection_2d.yolov5.post_processing_model import (
    unpack_observations,
)


class YoloEnsembleInferenceProvider:
    def __init__(self, model_path: str, input_shape: tuple, classes: dict):
        """
        Example yolo ensemble inference provider

        Args:
            model_path (str): the triton model name (unused)
            input_shape (tuple): the input shape of the model
            classes (dict): the class dict
        """

        # this is hardcoded for now
        # but will eventually be grabbed from the config
        self.model_name = "yolov5_ensemble"
        self.input_shape = input_shape

        triton_local_client = tritonclient.grpc.InferenceServerClient(
            url="127.0.0.1:8001", verbose=False, ssl=False
        )
        triton_local_client.name = "local_host"
        # double check that this ensemble model exists
        self.triton_client = triton_local_client

        model_metadata = self.triton_client.get_model_metadata(
            model_name=self.model_name, model_version="1"
        )
        self.classes = classes
        logger.info(
            f"Found ensemble model with metadata:\n {model_metadata.name}"
        )

    def process(self, batched_tensor: torch.Tensor) -> dict:
        """
        Generates the prediction of the model (end to end)

        Args:
            batched_tensor (torch.Tensor): the batch tensor

        Returns:
            dict: the dict of predictions (actor class to tensor prediction)
        """
        return run_inference(
            ensemble_model_name=self.model_name,
            sample_image=batched_tensor,
            triton_local_client=self.triton_client,
            new_shape=self.input_shape,
            classes_d=self.classes,
        )


def run_inference(
    ensemble_model_name: str,
    sample_image: np.ndarray,
    triton_local_client: tritonclient.grpc.InferenceServerClient,
    new_shape: list,
    classes_d: dict,
    confidence_threshold_f: float = 0.001,
    nms_threshold_f: float = 0.7,
) -> dict:
    """
    Runs inference on the ensemble model

    Args:
        ensemble_model_name (str): the ensemble model name
        sample_image (np.ndarray): the sample image to run inference on
        triton_local_client (tritonclient.grpc.InferenceServerClient): the
                  grpc triton client
        new_shape (list): the list of shape for the yolo model
        classes_d (dict): the classes dict
        confidence_threshold_f (float, optional): confidence threshold
                             of the model. Defaults to 0.001.
        nms_threshold_f (float, optional): nms threshold. Defaults to 0.7.

    Returns:
        dict: the dict of predictions (actor class to tensor prediction)
    """
    sample_image = sample_image.squeeze(0)
    sample_image = torch.from_numpy(
        sample_image.squeeze().numpy().transpose(2, 0, 1)
    )
    sample_image = sample_image.unsqueeze(0)
    # does some stuff
    new_shape = torch.tensor(new_shape).unsqueeze(0)
    classes = torch.tensor([classes_d.keys()[:2]])
    confidence_threshold = torch.tensor([[confidence_threshold_f]])
    nms_threshold = torch.tensor([[nms_threshold_f]])
    inputs = [
        InferInput("INPUT_0", sample_image.size(), "UINT8"),
        InferInput("INPUT_1", new_shape.size(), "INT32"),
        InferInput("INPUT_2", classes.size(), "INT32"),
        InferInput("INPUT_3", confidence_threshold.size(), "FP16"),
        InferInput("INPUT_4", nms_threshold.size(), "FP16"),
    ]
    inputs[0].set_data_from_numpy(sample_image.numpy().astype(np.uint8))
    inputs[1].set_data_from_numpy(new_shape.numpy().astype(np.int32))
    inputs[2].set_data_from_numpy(classes.numpy().astype(np.int32))
    inputs[3].set_data_from_numpy(
        confidence_threshold.numpy().astype(np.float16)
    )
    inputs[4].set_data_from_numpy(nms_threshold.numpy().astype(np.float16))
    outputs = [
        InferRequestedOutput("output0"),
        InferRequestedOutput("output1"),
    ]
    results = triton_local_client.infer(
        model_name=ensemble_model_name,
        inputs=inputs,
        outputs=outputs,
    )
    final_outputs = [
        torch.from_numpy(np.array(results.as_numpy(output.name()), copy=True))
        for output in outputs
    ]
    num_predictions, observations = final_outputs
    return unpack_observations(observations, num_predictions, classes_d)
