#!/bin/bash
##
## Copyright 2020-2021 Voxel Labs, Inc.
## All rights reserved.
##
## This document may not be reproduced, republished, distributed, transmitted,
## displayed, broadcast or otherwise exploited in any manner without the express
## prior written permission of Voxel Labs, Inc. The receipt or possession of this
## document does not convey any rights to reproduce, disclose, or distribute its
## contents, or to manufacture, use, or sell anything that it may describe, in
## whole or in part.
##

# Run every 2 hours via sudo crontab:
#
# 0 */2 * * * /home/edge-admin/scripts/reboot-rtsp-server.sh

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

# Ensure log directory exists
mkdir -p /home/edge-admin/scripts/logs

# Restart Exacqvision RTSP server
systemctl restart rtsp-server
echo "$(timestamp) Restarted rtsp-server" >> /home/edge-admin/scripts/logs/reboot-rtsp-server.log

# Restart all running containers
docker restart $(docker ps -a -q)
echo "$(timestamp) Restarted Docker containers" >> /home/edge-admin/scripts/logs/reboot-rtsp-server.log
