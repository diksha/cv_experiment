import torch

from lib.ml.inference.backends.torchscript import TorchscriptBackend
from lib.ml.inference.backends.triton import TritonInferenceBackend
from lib.ml.inference.factory.base import (
    InferenceBackendType,
    InferenceProviderFactoryBase,
)
from lib.ml.inference.tasks.segmentation.unet_v1.inference_provider import (
    InferenceProvider,
)


class UnetV1InferenceProviderFactory(InferenceProviderFactoryBase):
    """Inference Provider Factory."""

    def get_inference_provider(
        self,
        model_path: str,
    ) -> InferenceProvider:
        """Get correct fully connected unet inference provider class
        Args:
            model_path (str): path to model
        Returns:
            InferenceProvider: unet v1 inference provider
        Raises:
            RuntimeError: type not valid inference provider type
        """

        def generate_torchscript_inference_backend() -> TorchscriptBackend:
            """Generates the torchscript inference backend for spills

            Returns:
                TorchscriptBackend: the generated provider
            """
            return TorchscriptBackend(
                jit_file_path=model_path,
                device=torch.device(self._device),
                dtypes=[torch.float],
            )

        def generate_triton_inference_backend() -> TritonInferenceBackend:
            """Generates the triton inference backend for spills

            Returns:
                TritonInferenceBackend: the generated provider
            """
            return TritonInferenceBackend(
                model_name=model_path, triton_server_url=self.triton_server_url
            )

        type_map = {
            InferenceBackendType.TORCHSCRIPT: generate_torchscript_inference_backend,
            InferenceBackendType.TRITON: generate_triton_inference_backend,
        }

        backend_generator = type_map.get(self._type)

        if backend_generator is None:
            raise RuntimeError(
                f"{self._type} not a valid inference backend for spills"
            )
        return InferenceProvider(backend=backend_generator())
