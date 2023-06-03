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
import glob

import cv2 as cv
import numpy as np

from core.calibration.calibration_store.calibration_video_reader import (
    CalibrationVideoReader,
)

CHESSBOARD = (9, 15)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
video_reader = CalibrationVideoReader()
out_cap = cv.VideoWriter(
    "calibrate_americold_modesto_F_dock_ch5.mp4", 0x7634706D, 30, (960, 480)
)
index = 0
for img in video_reader.read("americold/modesto/F_dock/ch5"):
    print("Processing img.....", index)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv2.imwrite("frame_" + str(index), img)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        print("Found corners")
        cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        out_cap.write(img)
out_cap.release()
