#!/bin/bash

set -euo pipefail

# download the bazel-remote arm binary
BAZEL_REMOTE_VERSION=2.3.9
BAZEL_REMOTE_URL="https://github.com/buchgr/bazel-remote/releases/download/v${BAZEL_REMOTE_VERSION}/bazel-remote-${BAZEL_REMOTE_VERSION}-linux-arm64"

wget -qO /tmp/bazel-remote "${BAZEL_REMOTE_URL}"

# make the bazel-remote user/group
sudo useradd -rUM -s /dev/null bazel-remote

# install nvme-cli which is used by our ephemeral disk mount script
sudo yum -y install nvme-cli

# install our scripts and bazel-remote
sudo install -m 755 /tmp/bazel-remote /usr/local/bin/bazel-remote
sudo install -m 755 /tmp/files/mount-ephemeral-disk.sh /usr/local/bin/mount-ephemeral-disk.sh
sudo install -m 644 /tmp/files/bazel-remote.service /etc/systemd/system/bazel-remote.service

# finally set bazel-remote to start on system startup
sudo systemctl daemon-reload
sudo systemctl enable bazel-remote.service