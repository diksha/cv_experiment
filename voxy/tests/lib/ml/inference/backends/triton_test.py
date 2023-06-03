import unittest

import numpy as np
import torch
from tritonclient.grpc import (
    InferInput,
    InferRequestedOutput,
    model_config_pb2,
)
from tritonclient.grpc.model_config_pb2 import ModelInput, ModelOutput

from lib.ml.inference.backends.triton import TritonInputType, TritonOutputType


class TestTritonInputOutputType(unittest.TestCase):
    def test_triton_input_type(self):
        triton_input = ModelInput(
            name="input",
            dims=[1, 3, 224, 224],
            data_type=model_config_pb2.DataType.TYPE_FP32,
        )
        triton_input_type = TritonInputType.from_triton_config(triton_input)
        self.assertTrue(triton_input_type.input_name == "input")
        self.assertTrue(triton_input_type.input_size == [1, 3, 224, 224])
        self.assertTrue(triton_input_type.input_format == "FP32")
        self.assertTrue(triton_input_type.input_dtype_numpy == np.float32)

        triton_input_tensor = torch.rand(1, 3, 224, 224)
        triton_input_types = [triton_input_type]
        infer_inputs = TritonInputType.to_infer_inputs(
            triton_input_types, [triton_input_tensor]
        )
        self.assertTrue(len(infer_inputs) == 1)
        self.assertTrue(infer_inputs[0].name() == "input")
        self.assertTrue(infer_inputs[0].shape() == [1, 3, 224, 224])
        self.assertTrue(infer_inputs[0].datatype() == "FP32")
        self.assertTrue(isinstance(infer_inputs[0], InferInput))

    def test_triton_output_type(self):
        triton_output = ModelOutput(
            name="output",
            dims=[1, 1000],
            data_type=model_config_pb2.DataType.TYPE_FP32,
        )
        triton_output_type = TritonOutputType.from_triton_config(triton_output)
        self.assertTrue(triton_output_type.output_name == "output")
        infer_requested_outputs = TritonOutputType.to_requested_outputs(
            [triton_output_type]
        )
        self.assertTrue(len(infer_requested_outputs) == 1)
        self.assertTrue(infer_requested_outputs[0].name() == "output")
        self.assertTrue(
            isinstance(infer_requested_outputs[0], InferRequestedOutput)
        )
