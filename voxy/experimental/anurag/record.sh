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


# Sleep randomly for 0 to 100 seconds to ensure that we don't start all streams together.
sleep ${RANDOM:0:2}s

mkdir -p /home/voxel/data/videos/$4/$5/$6
mkdir -p /home/voxel/data/logs/$4/$5/$6


ffmpeg -stimeout 100000000 -rtsp_transport tcp -i "rtsp://$1:$2@$3:554/cam/realmonitor?channel=1&subtype=0" \
-f segment -segment_atclocktime 1 -segment_time 3600 -segment_format mp4 -reset_timestamps 1 -strftime 1 \
-c copy -map 0 /home/voxel/data/videos/$4/$5/$6/%s.mp4 >> /home/voxel/data/logs/$4/$5/$6/$(date +%s).log
