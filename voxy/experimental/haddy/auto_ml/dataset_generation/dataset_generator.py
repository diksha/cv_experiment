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
import glob
import os

from core.auto_ml.dataset_generation.cvat_converter import CvatAutoMLConveter

TRAIN_SET_NAME = "forklift-train.csv"
TEST_SET_NAME = "forklift-test.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, default="data/labels/cvat/annotations/pits/batch1"
    )
    parser.add_argument(
        "--output_path", type=str, default="data/labels/auto_ml/pits/batch1"
    )
    return parser.parse_args()


def split_train_test(op_folder_path):
    total = len(os.listdir(ip_folder_path))
    # do a 80/20 split
    test = total / 5
    if test == 0:
        test = 1
    test_index = 0
    with open(os.path.join(op_folder_path, TEST_SET_NAME), "w+") as test_f:
        with open(os.path.join(op_folder_path, TRAIN_SET_NAME), "w+") as train_f:
            for f in os.listdir(op_folder_path):
                if test_index < test:
                    with open(os.path.join(op_folder_path, f)) as r:
                        for line in r:
                            test_f.write(r)
                    test_index = test_index + 1
                else:
                    with open(os.path.join(op_folder_path, f)) as r:
                        for line in r:
                            train_f.write(r)


if __name__ == "__main__":
    args = parse_args()
    ip_folder_path = os.path.join(
        os.environ["BUILD_WORKSPACE_DIRECTORY"], args.input_path
    )
    op_folder_path = os.path.join(
        os.environ["BUILD_WORKSPACE_DIRECTORY"], args.output_path
    )
    for f in os.listdir(ip_folder_path):
        video_uuid = str(os.path.basename(f).replace(".xml", ""))
        op_labels_path = os.path.join(
            os.environ["BUILD_WORKSPACE_DIRECTORY"],
            os.path.join(op_folder_path, "{}.csv".format(video_uuid)),
        )
        ip_labels_path = os.path.join(
            os.environ["BUILD_WORKSPACE_DIRECTORY"],
            os.path.join(ip_folder_path, "{}.xml".format(video_uuid)),
        )
        convertor = Convertor(video_uuid, ip_labels_path, op_labels_path)
        convertor.parse()
        convertor.dump()

    split_train_test(op_folder_path)
