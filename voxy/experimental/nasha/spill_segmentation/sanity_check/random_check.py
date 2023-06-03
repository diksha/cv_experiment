import argparse
import random
import tempfile

import cv2
import numpy as np
from loguru import logger

from core.utils.aws_utils import (
    download_to_file,
    glob_from_bucket,
    separate_bucket_from_relative_path,
    upload_cv2_imageobj_to_s3,
    upload_file,
)


class SanityChecker:
    def __init__(
        self,
        training_dataset_path: str,
        output_path: str,
        sample_fraction: int,
    ):
        """_summary_

        Args:
            training_dataset_path (str): A path to the complete dataset
        """
        self.training_dataset_path = training_dataset_path
        self.img_names = []
        self.sample_fraction = sample_fraction
        self.output_path = output_path

    def count_training_samples(self):
        """Count the spill training samples in a given s3 directory

        Returns:
            tuple: (total number of image files, total number of annotation files)
        """

        bucket, prefix = separate_bucket_from_relative_path(
            self.training_dataset_path
        )
        filenames = glob_from_bucket(bucket, prefix, extensions=(".png"))

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
                s3_path=f"{output_prefix}/log.txt",
            )

        logger.debug(
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
                    path=f'{self.output_path}/{img_name.replace("/","-")}',
                    image=img_annot,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=(
            "A path to s3 directory to store the output of the sanity check, for example "
            "s3://voxel-users/nasha/spill_segmentation/sanity_check"
        ),
    )
    args, _ = parser.parse_known_args()
    sanity_checker = SanityChecker(
        training_dataset_path="s3://voxel-users/common/spill_data",
        sample_fraction=0.01,
        output_path=args.output_path,
    )

    sanity_checker.count_training_samples()
    sanity_checker.random_upload_images()
