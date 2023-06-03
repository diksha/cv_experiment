import numpy as np
import tritonclient.grpc as grpcclient

triton_client = grpcclient.InferenceServerClient(
    url="localhost:8001", verbose=False, ssl=False
)

print("Triton server ready", triton_client.is_server_ready())

for model in triton_client.get_model_repository_index().models:
    for _ in range(50):
        inputs = []
        outputs = []
        inputs.append(
            grpcclient.InferInput("images", [4, 3, 224, 224], "FP32")
        )

        input0_data = np.ones([4, 3, 224, 224], dtype=np.float32)
        inputs[0].set_data_from_numpy(input0_data)

        outputs.append(grpcclient.InferRequestedOutput("output0"))

        results = triton_client.infer(
            model_name=model.name,
            inputs=inputs,
            outputs=outputs,
        )

        out = results.as_numpy("output0")
        print(f"model_name={model.name} shape={out.shape}")
