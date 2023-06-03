#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import ffmpeg

from core.structs.attributes import RectangleXYWH


def crop_video(input_video: str, output_video: str, rectangle: RectangleXYWH):
    """
    Crops video from a input video path and an output video path given a rectangle polygon

    Args:
        input_video (str): the input video
        output_video (str): the output video path
        rectangle (RectangleXYWH): the rectangle region of interest to crop
    """
    # TODO: replace this with with pyav

    input_file_probe = ffmpeg.probe(filename=input_video)
    # get timebase
    timebase = int(input_file_probe["streams"][0]["time_base"].split("/")[-1])
    (
        ffmpeg.input(input_video)
        .crop(*rectangle.to_list())
        .output(output_video, vsync=0, video_track_timescale=timebase)
        .run(overwrite_output=True)
    )
