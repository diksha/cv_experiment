#!/bin/bash

set -euo pipefail

# find the ephemeral disk
EPHEMERAL_DISK=$(sudo nvme list | grep 'Amazon EC2 NVMe Instance Storage' | awk '{ print $1 }')

# do nothing if it is already mounted
if grep -qs "${EPHEMERAL_DISK}" /proc/mounts; then
    exit 0
fi

# reformat and mount it otherwise
mkdir -p /mnt/ephemeral
mkfs.xfs "${EPHEMERAL_DISK}"
mount -t xfs "${EPHEMERAL_DISK}" /mnt/ephemeral

mkdir -p /mnt/ephemeral/bazel-remote
sudo chown bazel-remote:bazel-remote /mnt/ephemeral/bazel-remote