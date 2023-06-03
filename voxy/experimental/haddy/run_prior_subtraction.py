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

import cv2
import numpy as np


class PriorSubtractor:
    def __init__(self, video_path, prior_path):
        self.video_path = video_path
        self.vdo = cv2.VideoCapture()
        self.video_uuid = str(os.path.basename(video_path).replace(".mp4", ""))
        self.prior_image = cv2.imread(prior_path)
        self.prior_image = cv2.cvtColor(self.prior_image, cv2.COLOR_BGR2HLS)

    def subtract_prior(self, image):
        assert self.prior_image.shape[0] == image.shape[0]
        assert self.prior_image.shape[1] == image.shape[1]
        h = self.prior_image.shape[0]
        w = self.prior_image.shape[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        mask = np.abs(image[:, :, 0] - self.prior_image[:, :, 0]) < 50
        image[mask, 0] = 0
        image[mask, 1] = 0
        mask = np.abs(image[:, :, 2] - self.prior_image[:, :, 2]) < 50
        image[mask, 2] = 0
        image[mask, 1] = 0
        # for y in range(0, h):
        #     for x in range(0, w):
        #         for f in range(0, 3):
        #             if f==0 and abs(image[y][x][f] - self.prior_image[y][x][f]) < 20:
        #                 image[y][x][1] = 0
        #                 image[y][x][f] = 0
        #             if f==2 and abs(image[y][x][f] - self.prior_image[y][x][f]) < 20:
        #                 image[y][x][1] = 0
        #                 image[y][x][f] = 0
        subtracted_image = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
        return subtracted_image

    def run(self):
        self.vdo.open(self.video_path)
        self.video_writer = cv2.VideoWriter(
            "prior_subraction.mp4",
            0x7634706D,
            10,
            (int(self.vdo.get(3)), int(self.vdo.get(4))),
        )
        while self.vdo.grab():
            _, img = self.vdo.retrieve()
            subtracted_image = self.subtract_prior(img)
            self.video_writer.write(subtracted_image)
        self.vdo.release()
        self.video_writer.release()


def parse_args():
    default_prior_path = os.path.join(
        os.environ["BUILD_WORKSPACE_DIRECTORY"], "data/front_view_pao_gym.png"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument(
        "--prior_path", type=str, required=False, default=default_prior_path
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prior_subtrator = PriorSubtractor(args.VIDEO_PATH, args.prior_path)
    prior_subtrator.run()
