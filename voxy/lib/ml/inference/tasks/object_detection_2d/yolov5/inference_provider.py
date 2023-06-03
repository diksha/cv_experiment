from typing import Callable, Dict, List

import torch

from lib.ml.inference.backends.base import InferenceBackend
from lib.ml.inference.provider.interface import AbstractInferenceProvider
from lib.ml.inference.tasks.object_detection_2d.yolov5.post_processing_model import (
    transform_and_post_process,
    unpack_observations,
)
from lib.ml.inference.tasks.object_detection_2d.yolov5.pre_processing_model import (
    preprocess_image,
)


class InferenceProvider(AbstractInferenceProvider):
    """TorchScript Inference Provider."""

    _confidence_threshold = torch.tensor([[0.001]])
    _nms_threshold = torch.tensor([[0.7]])

    def __init__(
        self,
        backend: InferenceBackend,
        input_shape: tuple,
        class_map: dict,
        io_processing: bool,
    ) -> None:
        """Constructor

        Args:
            jit_file_path (str): Path to torchscript file.
            input_shape (tuple): input shape which the model supports and requird for preprocessing.
            classes (dict): classes that are required for post processing.
            io_processing (bool): Whether or not the inference pipeline requires IO processing.
        """
        self.backend = backend
        self.input_shape = torch.tensor(input_shape).unsqueeze(0)
        self.class_map = class_map
        self._class_tensor = torch.tensor(class_map.keys()).unsqueeze(0)
        self._inference_pipeline = self._construct_inference_pipeline(
            io_processing
        )

    def process(self, batched_input: torch.tensor) -> list:
        """Takes and NHWC inputs and generates inference output.
        Change it to accept batched images.

        Generates a set of predictions by observation class. Format is:
        {
            CategoryType: torch.Tensor,
            ...
        }
        The format of the tensor is [bbox predictions | bbox confidence | class confidences]
        (typical yolo format)


        Args:
            batched_input (torch.tensor): raw NHWC input (batch, height, width, channels)

        Returns:
            list: set of predictions as indexed by their category indexed by their image
        """
        return self._inference_pipeline(batched_input)

    def _construct_inference_pipeline(
        self, requires_io_processing: bool
    ) -> Callable:
        """Constructs the inference pipeline.

        Args:
            requires_io_processing (bool): Whether or not the inference pipeline
                requires IO processing.

        Returns:
            typing.Callable: The inference pipeline.
        """

        def io_processing_pipeline(
            batched_image: torch.Tensor,
        ) -> List[Dict[str, torch.Tensor]]:
            """Inference pipeline with IO processing steps baked in (non ensemble use case)
            Args:
                batched_image (torch.Tensor): batched images in NHWC format
            Returns:
                typing.List[typing.Dict[str, torch.Tensor]]: a list of dictionaries of
                    detections by class
            """
            preprocessed_batch, offset, scale = preprocess_image(
                batched_image, self.input_shape
            )
            predictions = self.backend.infer([preprocessed_batch])[0]
            num_predictions, observations = transform_and_post_process(
                predictions,
                offset,
                scale,
                self._class_tensor,
                self._confidence_threshold,
                self._nms_threshold,
            )
            return unpack_observations(
                observations, num_predictions, self.class_map
            )

        def ensemble_pipeline(
            batched_image: torch.Tensor,
        ) -> List[Dict[str, torch.Tensor]]:
            """Inference pipeline with IO processing running in backend (ensemble use case)
            Args:
                batched_image (torch.Tensor): batched images in NHWC format
            Returns:
                typing.List[typing.Dict[str, torch.Tensor]]: a list of dictionaries of
                    detections by class
            """
            num_predictions, observations = self.backend.infer(
                [
                    batched_image,
                    self.input_shape,
                    self._class_tensor,
                    self._confidence_threshold,
                    self._nms_threshold,
                ]
            )
            return unpack_observations(
                observations, num_predictions, self.class_map
            )

        if requires_io_processing:
            return io_processing_pipeline
        return ensemble_pipeline
