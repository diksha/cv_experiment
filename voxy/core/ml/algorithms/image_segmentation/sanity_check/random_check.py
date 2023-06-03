import os
import random
import tempfile

import cv2
import numpy as np
import yaml
from loguru import logger

from core.utils.aws_utils import (
    download_to_file,
    glob_from_bucket,
    separate_bucket_from_relative_path,
    upload_cv2_imageobj_to_s3,
    upload_file,
)

_SANITY_CHECK_DEFAULTS = dict(output_bucket="voxel-users")


class SanityChecker:
    def __init__(
        self,
        config_training,
        sample_fraction: int,
    ):
        """Initialize sanity check

        Args:
            config_training (str): Training details config
            sample_fraction (int): Fraction of training images to sample
        """
        self.config = config_training
        self.training_dataset_path = os.path.join(
            "s3://",
            config_training["data_extraction_bucket"],
            config_training["data_extraction_relative_path"],
        )
        self.img_names = []
        self.sample_fraction = sample_fraction
        self.output_path = os.path.join(
            "s3://",
            _SANITY_CHECK_DEFAULTS["output_bucket"],
            config_training["model_save_relative_path"],
            config_training["model_name"],
        )
        self.prefixes = config_training["train"]

    def save_training_config(self):
        """Save the training config to model path
        Returns:
            str: Output path of training config
        """
        with tempfile.NamedTemporaryFile(suffix=".yaml") as config:
            with open(config.name, "w", encoding="utf-8") as config_file:
                yaml.dump(self.config, config_file)
                (
                    output_bucket,
                    output_prefix,
                ) = separate_bucket_from_relative_path(self.output_path)
            upload_file(
                bucket=output_bucket,
                local_path=config.name,
                s3_path=f"{output_prefix}/sanity_check/training_config.yaml",
            )
        return f"{output_prefix}/sanity_check/training_config.yaml"

    def count_training_samples(self):
        """Count the spill training samples in a given s3 directory

        Returns:
            tuple: (total number of image files, total number of annotation files)
        """

        bucket, root_prefix = separate_bucket_from_relative_path(
            self.training_dataset_path
        )
        filenames = []
        for prefix in self.prefixes:
            filenames.extend(
                glob_from_bucket(
                    bucket,
                    os.path.join(root_prefix, prefix.strip("/img")),
                    extensions=(".png"),
                )
            )

        self.img_names = [
            img_name for img_name in filenames if "/img/" in img_name
        ]
        annotation_names = [
            annotation_name
            for annotation_name in filenames
            if "/annotation/" in annotation_name
        ]
        img_cnt, annot_cnt = len(self.img_names), len(annotation_names)
        with tempfile.NamedTemporaryFile(suffix=".txt") as log:
            with open(log.name, "w", encoding="utf-8") as log_file:
                log_file.writelines(
                    [
                        f"Total number of images is {img_cnt}\n",
                        f"Total number of annotations is {annot_cnt}",
                    ]
                )
                (
                    output_bucket,
                    output_prefix,
                ) = separate_bucket_from_relative_path(self.output_path)
            upload_file(
                bucket=output_bucket,
                local_path=log.name,
                s3_path=f"{output_prefix}/sanity_check/log.txt",
            )

        logger.info(
            f"Total number of images is {img_cnt} and total number of annotations is {annot_cnt} "
        )
        return img_cnt, annot_cnt

    def random_upload_images(self):
        """Upload a random sample of the images"""

        random_filenames = random.sample(
            self.img_names,
            int(len(self.img_names) * self.sample_fraction),
        )
        bucket, _ = separate_bucket_from_relative_path(
            self.training_dataset_path
        )
        for img_name in random_filenames:
            with tempfile.NamedTemporaryFile(
                suffix=".png"
            ) as img, tempfile.NamedTemporaryFile(suffix=".png") as annotation:
                download_to_file(bucket, img_name, img.name)
                image = cv2.imread(img.name)

                download_to_file(
                    bucket,
                    img_name.replace("/img/", "/annotation/"),
                    annotation.name,
                )
                annot = (cv2.imread(annotation.name)) * 255

                height, width, _ = image.shape
                if height > width:
                    img_annot = np.concatenate((image, annot), axis=1)

                else:
                    img_annot = np.concatenate((image, annot), axis=0)

                upload_cv2_imageobj_to_s3(
                    path=f'{self.output_path}/sanity_check/images/{img_name.replace("/","-")}',
                    image=img_annot,
                )
        logger.info("Sanity check image upload complete")
