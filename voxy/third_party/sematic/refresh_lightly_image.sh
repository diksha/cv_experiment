#!/bin/bash
# This script can be periodically executed to use the latest code from Lightly
# in the Sematic pipeline. Once you have executed it, you need to update the
# SHA of lightly_worker_repush in the WORKSPACE file using the sha that is shown
# after the push.
set -euo pipefail

echo "$1"
lightly_image=lightly/worker:$1
sematic_image=203670452561.dkr.ecr.us-west-2.amazonaws.com/sematic:lightly-base-$1
echo "$lightly_image"
echo "$sematic_image"
cd third_party/sematic || exit
docker build --build-arg lightly_image="$lightly_image" -t "$sematic_image" ./
../../tools/aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 203670452561.dkr.ecr.us-west-2.amazonaws.com
docker push "$sematic_image"
dockerSHA=$(docker inspect --format='{{index .RepoDigests 0}}' "$sematic_image" | perl -wnE'say /sha256.*/g')
echo "$dockerSHA"
echo "Please update WORKSPACE with following lightly_worker_repush info"
echo 'authenticated_container_pull(
    name = "lightly_worker_repush",
    digest = "'"${dockerSHA}"'",
    registry = "203670452561.dkr.ecr.us-west-2.amazonaws.com",
    repository = "sematic",
    tag = "lightly-base-'"${1}"'",
)'
echo 'For local runs also update core/ml/data/curation/lib/lightly_worker.py with tag '"$1"
