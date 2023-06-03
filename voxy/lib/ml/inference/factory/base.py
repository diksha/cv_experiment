from enum import Enum

import torch

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)


class InferenceBackendType(Enum):
    TORCHSCRIPT = 0
    TRT = 1
    TRITON = 2


class InferenceProviderFactoryBase:
    """Inference Provider Factory Base"""

    _K_DEVICE_GPU = "cuda:0"
    _K_DEVICE_CPU = "cpu"

    def __init__(
        self,
        local_inference_provider_type: InferenceBackendType,
        gpu_runtime: GpuRuntimeBackend,
        triton_server_url: str,
    ):
        """Constructor
        Args:
            local_inference_provider_type (str): inference provider type
            gpu_runtime (GpuRuntimeBackend): gpu runtime (local, remote)

        Raises:
            RuntimeError: input trt inference provider without CUDA available
        """
        self.triton_server_url = triton_server_url
        inference_provider_map = {
            GpuRuntimeBackend.GPU_RUNTIME_BACKEND_LOCAL: local_inference_provider_type,
            GpuRuntimeBackend.GPU_RUNTIME_BACKEND_REMOTE_TRITON: InferenceBackendType.TRITON,
        }
        inference_provider_type = inference_provider_map.get(gpu_runtime, None)
        self._type = inference_provider_type
        available_device = (
            self._K_DEVICE_GPU
            if torch.cuda.is_available()
            else self._K_DEVICE_CPU
        )
        if (
            self._type == InferenceBackendType.TRT
            and available_device != self._K_DEVICE_GPU
        ):
            raise RuntimeError(
                "TRT inference provider specified without CUDA available!"
            )
        device_map = {
            InferenceBackendType.TORCHSCRIPT: available_device,
            InferenceBackendType.TRT: self._K_DEVICE_GPU,
            InferenceBackendType.TRITON: self._K_DEVICE_CPU,
        }
        self._device = device_map[self._type]

    def get_device(self) -> torch.device:
        """
        Grabs and returns the device

        Returns:
            torch.device: the device used for this model
        """
        return self._device
