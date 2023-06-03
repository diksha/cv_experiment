#!/bin/bash

set -euo pipefail

BAZEL_REMOTE_VERSION=2.4.0
BAZEL_REMOTE_URL="https://github.com/buchgr/bazel-remote/releases/download/v$${BAZEL_REMOTE_VERSION}/bazel-remote-$${BAZEL_REMOTE_VERSION}-linux-x86_64"
BAZEL_REMOTE_SHA256="717a44dd526c574b0a0edda1159f5795cc1b2257db1d519280a3d7a9c5addde5"

# download bazel remote and verify the checksum
wget -qO /tmp/bazel-remote "$${BAZEL_REMOTE_URL}"
echo "$${BAZEL_REMOTE_SHA256}  /tmp/bazel-remote" | sha256sum -c

cat <<EOT >> /tmp/bazel-remote.service
# Example configuration for systemd based Linux machines.
#
# Customize this file and add it to /etc/systemd/system/
# then run "systemctl daemon-reload" so systemd finds the file.
# "systemctl start bazel-remote" will then start the service
# and "systemctl enable bazel-remote" will make systemd start
# the service after booting up.

[Unit]
Description=bazel-remote cache

[Service]
# Assuming you have created a bazel-remote user and group, that can write
# to the cache directory specified in ExecStart below:
User=bazel-remote
Group=bazel-remote

# We need to have a lot of files open at once.
LimitNOFILE=40000

# Try to avoid "runtime: failed to create new OS thread (have 2458 already; errno=11)"
# errors. You can check if this worked by running "systemctl status bazel-remote"
# and see if there's a "Tasks: 18 (limit: 2457)" line (hopefully not, after adding this).
LimitNPROC=infinity
TasksMax=infinity

Restart=on-failure

# If you want to log GC events.
#Environment=GODEBUG=gctrace=1

# If you want bazel-remote to listen on ports < 1024 make sure to run
# setcap cap_net_bind_service+eip /absolute/path/to/bazel-remote
# (as root) to allow the executable to have this capability without
# running as root.
ExecStart=/usr/local/bin/bazel-remote \
	--max_size 200 \
	--dir /tmp/bazel-remote-cache \
    --s3.endpoint s3.${AWS_REGION}.amazonaws.com \
    --s3.region ${AWS_REGION} \
    --s3.key_version 2 \
    --s3.bucket ${CACHE_BUCKET} \
    --s3.auth_method iam_role

PermissionsStartOnly=true

[Install]
WantedBy=multi-user.target
EOT

# make the bazel-remote user/group
sudo useradd -rUM -s /dev/null bazel-remote

# install our scripts and bazel-remote
sudo install -m 755 /tmp/bazel-remote /usr/local/bin/bazel-remote
sudo install -m 644 /tmp/bazel-remote.service /etc/systemd/system/bazel-remote.service

sudo mkdir -p /mnt/ephemeral/bazel-remote
sudo chown bazel-remote:bazel-remote /mnt/ephemeral/bazel-remote

# finally set bazel-remote to start on system startup
sudo systemctl daemon-reload
sudo systemctl enable bazel-remote.service
sudo systemctl start bazel-remote.service
