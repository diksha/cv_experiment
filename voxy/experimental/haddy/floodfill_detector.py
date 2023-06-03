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
import os

import cv2
from numpy import *

from core.cv.door.state_detector import DoorStateDetector

test_img_folder = "data/door_examples"

door_state_detector = DoorStateDetector()
for imgFile in os.listdir(
    os.path.join(os.environ["BUILD_WORKSPACE_DIRECTORY"], test_img_folder)
):
    imgFile = os.path.join(
        os.environ["BUILD_WORKSPACE_DIRECTORY"], test_img_folder, imgFile
    )
    if not imgFile.endswith("jpg"):
        continue
    img = cv2.imread(imgFile)
    height, width, channels = img.shape
    mask = zeros((height + 2, width + 2), uint8)

    # the starting pixel for the floodFill
    door_start_pixel_left = (370, 80)
    # maximum distance to start pixel:
    diff = (2, 2, 2)

    door_detector_prediction = door_state_detector("e_dock_north_ch12", img)

    retval, img, mask, rect = cv2.floodFill(
        img, mask, door_start_pixel_left, (0, 255, 0), diff, diff
    )

    print(retval)

    # check the size of the floodfilled area, if its large the door is closed:
    if retval > 10000:
        print(imgFile + ": door closed, " + door_detector_prediction)
    else:
        print(imgFile + ": door open, " + door_detector_prediction)

    cv2.imwrite(
        imgFile.replace(".jpg", "").replace("door_examples", "door_examples/results")
        + "_result.jpg",
        img,
    )
