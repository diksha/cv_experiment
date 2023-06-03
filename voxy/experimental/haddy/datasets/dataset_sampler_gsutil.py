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

"""Create a dataset that is a subset of the larger dataset"""
import argparse
import json
import os
import random
import secrets
import subprocess
import tempfile

import cv2
import yaml
from tqdm import tqdm

from core.infra.cloud.gcs_cv2_utils import upload_cv2_image_to_gcs
from core.infra.cloud.gcs_utils import (
    blob_count_with_prefix,
    download_blob,
    dump_to_gcs,
    list_blobs_with_prefix,
    read_from_gcs,
    separate_bucket_from_relative_path,
    upload_to_gcs,
)


class DatasetSampler:
    def __init__(
        self,
        dataset_config_path,
        output_path,
        dataset_root_dir=None,
        image_format=".png",
        max_samples=None,
        max_samples_from_one_source=20,
    ):
        self.dataset_config_path = dataset_config_path
        self.dataset_root_dir = (
            os.path.dirname(self.dataset_config_path)
            if dataset_root_dir is None
            else dataset_root_dir
        )
        self.output_path = output_path
        self.image_format = image_format
        self.max_samples = self.count() if not max_samples else max_samples
        self.max_samples_from_one_source = max_samples_from_one_source
        self.config = yaml.load(read_from_gcs(self.dataset_config_path))
        self.available_images_paths = self._get_available_images_paths()
        self.samples = 0

    def _get_available_images_paths(self):
        splits = ["train", "test", "val"]
        return [path for split in splits for path in self.config[split]]

    def count(self):
        if self.max_samples:
            return self.max_samples
        else:
            total_count = 0
            for images_path in self.available_images_paths:
                images_path = images_path.replace(
                    "/data", self.dataset_root_dir
                )
                labels_path = os.path.join(
                    os.path.dirname(images_path), "labels"
                )
                (
                    labels_bucket,
                    path,
                ) = separate_bucket_from_relative_path(labels_path)
                total_count = total_count + blob_count_with_prefix(
                    labels_bucket,
                    prefix=labels_path.replace(f"gs://{labels_bucket}/", ""),
                )
            print(total_count)
            return total_count

    def sample(self):
        # Randomly select a label path, fetch 20 samples from it and repeat the process
        # till max_samples are obtained. Copy labels and corresponding images to
        # given output folder.
        output_images_dir_path = os.path.join(self.output_path, "images")
        output_labels_dir_path = os.path.join(self.output_path, "labels")
        with tqdm(total=self.max_samples) as progress_bar:
            while self.samples < self.max_samples:
                selected_images_path = random.choice(
                    self.available_images_paths
                )
                if "real-world-ppe" in selected_images_path:
                    # skip real world data
                    break
                images_path = selected_images_path.replace(
                    "/data", self.dataset_root_dir
                )
                labels_path = images_path.replace("images", "labels")
                (
                    labels_bucket,
                    path,
                ) = separate_bucket_from_relative_path(labels_path)
                labels_blobs = list_blobs_with_prefix(
                    labels_bucket,
                    prefix=labels_path.replace(f"gs://{labels_bucket}/", ""),
                )
                sampled_from_source = 0
                salt = secrets.token_hex(4)
                for blob_name in labels_blobs:
                    if sampled_from_source >= self.max_samples_from_one_source:
                        break
                    label_path = os.path.join(
                        labels_path, os.path.basename(blob_name)
                    )
                    image_path = os.path.join(
                        images_path,
                        os.path.basename(blob_name).replace(
                            ".txt", self.image_format
                        ),
                    )
                    # image
                    image_output_path = os.path.join(
                        output_images_dir_path,
                        os.path.basename(blob_name).replace(
                            ".txt", "_" + salt + self.image_format
                        ),
                    )
                    input_path = image_path
                    output_path = image_output_path
                    file_copy_command_template = (
                        f"./tools/gsutil -m cp {input_path} {output_path}"
                    )
                    image_status = subprocess.run(
                        file_copy_command_template.split(),
                        cwd=os.environ["BUILD_WORKSPACE_DIRECTORY"],
                        env=dict(
                            os.environ,
                        ),
                    )

                    # label
                    label_output_path = os.path.join(
                        output_labels_dir_path,
                        os.path.basename(blob_name).replace(
                            ".txt", "_" + salt + ".txt"
                        ),
                    )
                    input_path = label_path
                    output_path = label_output_path
                    file_copy_command_template = (
                        f"./tools/gsutil -m cp {input_path} {output_path}"
                    )
                    label_status = subprocess.run(
                        file_copy_command_template.split(),
                        cwd=os.environ["BUILD_WORKSPACE_DIRECTORY"],
                        env=dict(
                            os.environ,
                            PATH=f"{os.environ['PYTHONHOME']}/../..:{os.environ['PATH']}",
                        ),
                    )

                    if (
                        not label_status.returncode
                        and not image_status.returncode
                    ):
                        sampled_from_source = sampled_from_source + 1
                        progress_bar.update(1)
                self.samples = self.samples + sampled_from_source


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config_path", "-c", type=str, required=True)
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        required=True,
    )
    parser.add_argument("--max_samples", "-n", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # dataset_sampler = DatasetSampler(
    #     dataset_config_path="gs://voxel-datasets/derived/aimh.isti.cnr.it/dataset.yml",
    #     output_path="gs://voxel-users/common/ppe_synthetic_data_derived_5k/",
    #     max_samples=5000,
    # )

    dataset_sampler = DatasetSampler(
        dataset_config_path=args.dataset_config_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
    )

    dataset_sampler.sample()
