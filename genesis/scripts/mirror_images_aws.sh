#!/bin/bash
set -euo pipefail
TARGET_REGISTRY="public.ecr.aws/voxelai"

aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "${TARGET_REGISTRY}"

IMAGE_TO_CLONE="${1:?'Provider image to clone as first param'}"
TARGET_TAG="${2:-"${IMAGE_TO_CLONE}"}"


if [ -n "${SOURCE_REGISTRY:-}" ]; then
  docker pull --platform=linux/amd64 "${SOURCE_REGISTRY}/${IMAGE_TO_CLONE}"
  docker tag "${SOURCE_REGISTRY}/${IMAGE_TO_CLONE}" "${TARGET_REGISTRY}/${IMAGE_TO_CLONE}"
  docker push "${TARGET_REGISTRY}/${IMAGE_TO_CLONE}"
else
  docker pull --platform=linux/amd64 "${IMAGE_TO_CLONE}"
  docker tag "${IMAGE_TO_CLONE}" "${TARGET_REGISTRY}/${TARGET_TAG}"
  docker push "${TARGET_REGISTRY}/${TARGET_TAG}"
fi

docker pull "${IMAGE_TO_CLONE}" &>/dev/null || : 

