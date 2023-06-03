#!/usr/bin/env python
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
import argparse
import os
import pickle
from glob import glob

import cv2
import numpy as np
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate camera using a video of a chessboard or a sequence of images."
    )
    parser.add_argument("input", help="input video file or glob mask")
    parser.add_argument("out", help="output calibration yaml file")
    parser.add_argument(
        "--debug-dir",
        help="path to directory where images with detected chessboard will be written",
        default=None,
    )
    parser.add_argument("-c", "--corners", help="output corners file", default=None)
    parser.add_argument(
        "-fs",
        "--framestep",
        help="use every nth frame in the video",
        default=20,
        type=int,
    )
    # parser.add_argument('--figure', help='saved visualization name', default=None)
    args = parser.parse_args()

    if "*" in args.input:
        source = glob(args.input)
    else:
        source = cv2.VideoCapture(args.input)
    # square_size = float(args.get('--square_size', 1.0))

    pattern_size = (5, 15)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    # pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    i = -1
    while True:
        i += 1
        if isinstance(source, list):
            # glob
            if i == len(source):
                break
            img = cv2.imread(source[i])
        else:
            # cv2.VideoCapture
            retval, img = source.read()
            if not retval:
                break
            if i % args.framestep != 0:
                continue

        if i < 770 or i > 840:
            continue
        print("Searching for chessboard in frame " + str(i) + "..."),
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(
            img, pattern_size, flags=cv2.CALIB_CB_FILTER_QUADS
        )
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        if found and args.debug_dir:
            img_chess = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(img_chess, pattern_size, corners, found)
            cv2.imwrite(os.path.join(args.debug_dir, "%04d.png" % i), img_chess)
        if not found:
            continue
        img_points.append(corners)
        obj_points.append(pattern_points)
        print("FOUND")

    if args.corners:
        with open(args.corners, "wb") as fw:
            pickle.dump(img_points, fw)
            pickle.dump(obj_points, fw)
            pickle.dump((w, h), fw)

    # load corners
    #    with open('corners.pkl', 'rb') as fr:
    #        img_points = pickle.load(fr)
    #        obj_points = pickle.load(fr)
    #        w, h = pickle.load(fr)

    print("\nPerforming calibration...")
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )
    rvecs = np.asarray(rvecs)
    tvecs = np.asarray(tvecs)
    print("RMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs)
    print("rvecs: ", rvecs)
    print("tvecs: ", tvecs)

    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, jacobian = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs
        )
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error / len(obj_points))

    # # fisheye calibration
    # rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.fisheye.calibrate(
    #     obj_points, img_points,
    #     (w, h), camera_matrix, np.array([0., 0., 0., 0.]),
    #     None, None,
    #     cv2.fisheye.CALIB_USE_INTRINSIC_GUESS, (3, 1, 1e-6))
    # print "RMS:", rms
    # print "camera matrix:\n", camera_matrix
    # print "distortion coefficients: ", dist_coefs.ravel()

    calibration = {
        "rms": rms,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coefs": dist_coefs.tolist(),
        "rvecs": rvecs.tolist(),
        "tvecs": tvecs.tolist(),
    }
    with open(args.out, "w") as fw:
        yaml.dump(calibration, fw)

    # Take an Object Point and See if conversion to ImgPoint and back to
    # ObjectPoint returns similar point.
    test_obj_point = np.asarray([5, 5, 0], dtype=np.float32)
    test_img_point, _ = cv2.projectPoints(
        test_obj_point, rvecs[0], tvecs[0], camera_matrix, dist_coefs
    )

    # projection matrix
    Lcam = camera_matrix.dot(np.hstack((cv2.Rodrigues(rvecs[0])[0], tvecs[0])))
    px = test_img_point[0][0][0]
    py = test_img_point[0][0][1]
    Z = 0
    X = np.linalg.inv(
        np.hstack((Lcam[:, 0:2], np.array([[-1 * px], [-1 * py], [-1]])))
    ).dot(-Z * Lcam[:, 2] - Lcam[:, 3])
    print(X.tolist()[:2])
