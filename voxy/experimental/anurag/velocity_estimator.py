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
import collections
import os

import cv2
import numpy as np
import yaml

from core.structs.attributes import RectangleXYXY


class ForkliftVelocityEstimator:
    def __init__(self):
        path = os.path.join(
            os.environ["BUILD_WORKSPACE_DIRECTORY"],
            "data/artifacts/calibration/edock_ch22.yaml",
        )
        with open(path, "r") as stream:
            calibrations = yaml.load(stream)

        self.camera_matrix = np.asarray(calibrations["camera_matrix"])
        self.dist_coefs = np.asarray(calibrations["dist_coefs"])
        self.rvecs = np.asarray(calibrations["rvecs"])
        self.tvecs = np.asarray(calibrations["tvecs"])
        self.Lcam = self.camera_matrix.dot(
            np.hstack((cv2.Rodrigues(self.rvecs[0])[0], self.tvecs[0]))
        )

        self.track_map = collections.defaultdict(list)

    def __call__(self, actors, timestamp):
        for actor in actors:
            rect = RectangleXYXY.from_polygon(actor.polygon)
            px, py, Z = rect.bottom_right_vertice.x, rect.bottom_right_vertice.y, 0
            X = np.linalg.inv(
                np.hstack((self.Lcam[:, 0:2], np.array([[-1 * px], [-1 * py], [-1]])))
            ).dot(-Z * self.Lcam[:, 2] - self.Lcam[:, 3])
            self.track_map[actor.track_id].append(X.tolist()[:2])

        # print(self.track_map)
        return
