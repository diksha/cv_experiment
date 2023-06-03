import typing
from collections import OrderedDict, namedtuple

import tensorrt as trt
import torch

from core.structs.model import ModelConfiguration
from lib.ml.inference.backends.base import InferenceBackend

Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))


class TRTBackend(InferenceBackend):
    """TRTBackend.
    Check if we can use https://github.com/NVIDIA-AI-IOT/torch2trt
    """

    def __init__(self, engine_file_path: str, device: torch.device) -> None:
        """Constructor.

        Add input and output names.

        Args:
            engine_file_path (str): Path to enginer file.
            device (torch.device): Device whether cuda 0 or another gpu.
        """

        self.device = device

        with open(engine_file_path, "rb") as file_pointer, trt.Runtime(
            trt.Logger(trt.Logger.INFO)
        ) as runtime:
            self.model = runtime.deserialize_cuda_engine(file_pointer.read())

        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        self.dynamic = False
        self.dtypes = []

        output_names = []
        input_names = []
        input_is_dynamic = []
        for index in range(self.model.num_bindings):
            name = self.model.get_binding_name(index)
            dtype = self.trt_dtype_to_torch(
                self.model.get_binding_dtype(index)
            )
            if self.model.binding_is_input(index):
                if -1 in tuple(self.model.get_binding_shape(index)):
                    self.context.set_binding_shape(
                        index, tuple(self.model.get_profile_shape(0, index)[2])
                    )
                    input_is_dynamic.append(True)
                else:
                    input_is_dynamic.append(False)
                input_names.append(name)
            else:
                output_names.append(name)

            shape = tuple(self.model.get_binding_shape(index))
            data = torch.zeros(
                (1, *shape[1:]),
                dtype=dtype,
                device=torch.device("cuda:0"),
            ).contiguous()
            self.bindings[name] = Binding(
                name, dtype, shape, data, int(data.data_ptr())
            )

        self.binding_addrs = OrderedDict(
            (name, value.ptr) for name, value in self.bindings.items()
        )
        self.output_names = sorted(output_names)
        self.input_names = input_names
        self.input_is_dynamic = input_is_dynamic

    def trt_dtype_to_torch(self, dtype: trt.DataType) -> torch.dtype:
        """Type Convertor.

        Args:
            dtype (trt.DataType): Input trt DataType.

        Raises:
            RuntimeError: Unsupported dtype.

        Returns:
            torch.dtype: Converted torch dtype.
        """
        if dtype == trt.float16:
            return torch.float16

        if dtype == trt.float32:
            return torch.float32

        raise RuntimeError(f"Unsupported dtype : {dtype}")

    def get_config(self) -> ModelConfiguration:
        """
        Loads a model configuration relating to all relevant preprocessing
        postprocessing for the model

        This is related to how to configure preprocessing/post processing
        since that may change from model to model
        (different transforms, etc.)

        Raises:
           RuntimeError: As the method is not implemented
        """
        # TODO: update this to grab the proper model configuration
        raise RuntimeError("Not Implemented!")

    def infer(
        self, input_tensors: typing.List[torch.Tensor]
    ) -> typing.List[torch.Tensor]:
        """Infer function that takes an input runs the model on it and return output predictions.

        Args:
            input_tensors (typing.List[torch.Tensor]): input to the model,
                         in case of image generally NCHW.

        Raises:
            RuntimeError: If input and output shape mismatches.
            ValueError: if the number of input tensors is not equal to the number of input names.

        Returns:
            torch.Tensor: Ouputs generate by the model inference.
        """

        # images is in the self.input_names
        if len(input_tensors) != len(self.input_names):
            raise ValueError(
                f"Expected {len(self.input_names)} input tensors, but got {len(input_tensors)}"
            )
        for is_dynamic, input_name, input_tensor in zip(
            self.input_is_dynamic, self.input_names, input_tensors
        ):
            if is_dynamic:
                index = self.model.get_binding_index(input_name)
                self.context.set_binding_shape(index, input_tensor.shape)
                self.bindings[input_name] = self.bindings[input_name]._replace(
                    shape=input_tensor.shape
                )
        for name in self.output_names:
            index = self.model.get_binding_index(name)
            self.bindings[name].data.resize_(
                tuple(self.context.get_binding_shape(index))
            )

        for input_name, input_tensor in zip(self.input_names, input_tensors):
            binding = self.bindings[input_name]
            input_tensor_t = (
                input_tensor.to(binding.dtype).to(self.device).contiguous()
            )
            self.binding_addrs[input_name] = int(input_tensor_t.data_ptr())

        self.context.execute_v2(list(self.binding_addrs.values()))

        return [
            binding.data
            for name, binding in self.bindings.items()
            if name in self.output_names
        ]
