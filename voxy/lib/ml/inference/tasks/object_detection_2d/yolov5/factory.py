import torch

from lib.ml.inference.backends.torchscript import TorchscriptBackend
from lib.ml.inference.backends.triton import TritonInferenceBackend
from lib.ml.inference.backends.trt import TRTBackend
from lib.ml.inference.factory.base import (
    InferenceBackendType,
    InferenceProviderFactoryBase,
)
from lib.ml.inference.tasks.object_detection_2d.yolov5.inference_provider import (
    InferenceProvider,
)


class InferenceProviderFactory(InferenceProviderFactoryBase):
    """Inference Provider Factory."""

    _should_process_io = True

    def get_inference_provider(
        self,
        model_path: str,
        input_shape: tuple,
        class_map: dict,
    ) -> InferenceProvider:
        """Get correct yolo inference provider class
        Args:
            model_path (str): path to model
            input_shape (str): image input shape for YOLOv5 strides
            class_map (dict): dictionary of classes to detect
        Returns:
            InferenceProvider: Yolov5 inference provider
        Raises:
            RuntimeError: type not valid inference provider type
        """

        def generate_torchscript_inference_backend() -> TorchscriptBackend:
            """Generate torchscript inference backend for yolov5
            Returns:
                TorchscriptBackend
            """
            return TorchscriptBackend(
                jit_file_path=model_path,
                device=torch.device(self._device),
                dtypes=[torch.half],
            )

        def generate_trt_inference_backend() -> TRTBackend:
            """Generate trt inference backend for yolov5
            Returns:
                TRTBackend
            """
            return TRTBackend(
                engine_file_path=model_path,
                device=torch.device(self._device),
            )

        def generate_triton_inference_backend() -> TritonInferenceBackend:
            """Generate triton inference backend for yolov5
            Returns:
                TritonInferenceBackend
            """
            self._should_process_io = False
            return TritonInferenceBackend(
                model_name=model_path,
                triton_server_url=self.triton_server_url,
                is_ensemble=True,
            )

        type_map = {
            InferenceBackendType.TORCHSCRIPT: generate_torchscript_inference_backend,
            InferenceBackendType.TRT: generate_trt_inference_backend,
            InferenceBackendType.TRITON: generate_triton_inference_backend,
        }

        backend_generator = type_map.get(self._type)
        if backend_generator is None:
            raise RuntimeError(
                f"{self._type} not a valid inference backend for YOLOv5"
            )
        return InferenceProvider(
            backend=backend_generator(),
            input_shape=input_shape,
            class_map=class_map,
            io_processing=self._should_process_io,
        )
