#!/bin/bash
# This script can be periodically executed to use the latest code from YOLO
# in the Sematic pipeline. Once you have executed it, you need to update the
# SHA of yolov5_repush in the WORKSPACE file using the sha that is shown
# after the push.
set -euo pipefail

cd third_party/sematic || exit
date_utc=$(date --utc +'%Y-%m-%dT%H-%M-%S')
yolo_sematic_image=203670452561.dkr.ecr.us-west-2.amazonaws.com/sematic:yolov5-base-$date_utc
echo "$yolo_sematic_image"
docker build -t "$yolo_sematic_image" -f yolov5.Dockerfile .
docker push "$yolo_sematic_image"
dockerSHA=$(docker inspect --format='{{index .RepoDigests 0}}' "$yolo_sematic_image" | perl -wnE'say /sha256.*/g')
echo "$dockerSHA"
echo "Please update WORKSPACE with following lightly_worker_repush info"
echo 'authenticated_container_pull(
    name = "yolov5_repush",
    digest = "'"${dockerSHA}"'",
    registry = "203670452561.dkr.ecr.us-west-2.amazonaws.com",
    repository = "sematic",
    tag = "yolov5-base-'"${date_utc}"'",
)'
