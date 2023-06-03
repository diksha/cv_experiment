# Experimental triton model repo tool

This tool allows for a triton model repository to be created either in a local directory or in S3. The macro in `defs.bzl` can be used to create a `bazel run` target. See the example yaml files in this directory as well as `repo.proto` examples and the configuration schema. The same configuration file may be re-used to construct repositories in multiple places by re-using the file in multiple `triton_model_repository` macros.

## Ensemble Experiment

To build the ensemble experiment locally:

`bazel run :ensemble-experiment.update`

To build the ensemble experiment in s3:

`bazel run :ensemble-experiment-s3.update`

To run the ensemble experiment locally:

`docker run -v /tmp/modelrepos:/tmp/modelrepos --gpus 0 --rm -p8000:8000 -p8001:8001 -p8002:8002 nvcr.io/nvidia/tritonserver:22.07-py3 tritonserver --model-repository=/tmp/modelrepos/ensemble-experiment --model-control-mode=none --disable-auto-complete-config`

To run the ensemble experiment locally, but using the s3 model repo, you will need to give the docker container access to your s3 credentials by mounting the credentials to the docker container. Update the following command to direct the mount to your .aws folder

`docker run -v /PATH/TO/.aws:/root/.aws --gpus 0 --rm -p8000:8000 -p8001:8001 -p8002:8002 nvcr.io/nvidia/tritonserver:22.07-py3 tritonserver --model-repository=s3://voxel-experimental-triton-models/ensemble-experiment --model-control-mode=none --disable-auto-complete-config`

To run the ensemble experiment from a remote triton, use the following triton server address:

`k8s-tritonex-tritonex-e166009595-c1e917fd744043c9.elb.us-west-2.amazonaws.com:8001`

## Example

To build `jorge-experiment.yaml` run the local bazel target like this `bazel run :jorge-experiment.update`. Triton can then be started with the following command: `docker run -v /tmp/modelrepos:/tmp/modelrepos --gpus 0 --rm -p8000:8000 -p8001:8001 -p8002:8002 nvcr.io/nvidia/tritonserver:22.07-py3 tritonserver --model-repository=/tmp/modelrepos/jorge-experiment --model-control-mode=none --disable-auto-complete-config`
