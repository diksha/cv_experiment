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
"""
Arguments:
one of images_path or image_dir: gcs uri, s3 uri or a local path
image_collection_name: defaults to uuid
src: source of files, gcs, s3 or local
camera_uuid or synthetic_folder: camera_uuid or the synthetic_folder the list of images belong to

Example Usage:
1. ./bazel run -- core/labeling/voxel_logs_store:ingest_image_collection
--images_dir s3://voxel-lightly-output/spills/uscold/laredo/dock01/cha/.lightly/frames/
--src s3 --camera_uuid verst/walton/0001/cha
2. ./bazel run -- core/labeling/voxel_logs_store:ingest_image_collection
--src gcs --camera_uuid verst/walton/0001/cha
--images_path s3://voxel-lightly-output/spills/uscold/laredo/dock01/cha/.lightly/frames/1.jpg
s3://voxel-lightly-output/spills/uscold/laredo/dock01/cha/.lightly/frames/2.jpg
"""
import argparse
import os
import uuid
from enum import Enum, unique
from typing import List, Optional

import sematic
from loguru import logger
from sematic.resolvers.silent_resolver import SilentResolver

from core.infra.cloud.gcs_utils import (
    copy_blob,
    get_files_in_bucket,
    separate_bucket_from_relative_path,
    upload_to_gcs,
)
from core.labeling.logs_store.ingest_data_collection_metaverse import (
    DataCollectionInfo,
    ingest_data_collections_to_metaverse,
)
from core.structs.data_collection import DataCollectionType
from core.utils.aws_utils import (
    copy_object,
    get_bucket_path_from_s3_uri,
    glob_from_bucket,
)


@unique
class ImagesSource(Enum):
    """Source of the images"""

    UNKNOWN = 0
    S3 = 1
    GCS = 2
    LOCAL = 3


def copy_s3_images_to_voxel_logs(
    image_collection_name: str,
    output_images_folder: str,
    images_path: List = None,
    images_dir: str = None,
) -> List:
    """Copy s3 images to voxel logs

    Args:
        image_collection_name (str): collection name
        output_images_folder (str): folder to store images in
        images_path (List, optional): path of images. Defaults to None.
        images_dir (str, optional): directory of images. Defaults to None.

    Returns:
        List: s3 path images were uploaded to
    """
    if images_dir:
        bucket, s3_path = get_bucket_path_from_s3_uri(images_dir)
        images_path = [
            os.path.join(f"s3://{bucket}", f)
            for f in glob_from_bucket(bucket, s3_path, ("jpg", "png"))
        ]
    uploaded_images = []
    for image_path in images_path:
        bucket, s3_path = get_bucket_path_from_s3_uri(image_path)
        filename = os.path.basename(s3_path)
        dest_path = os.path.join(
            "s3://voxel-logs",
            output_images_folder,
            image_collection_name,
            filename,
        )
        copy_object(image_path, dest_path)
        uploaded_images.append(dest_path)
    return uploaded_images


def upload_local_images_gcs(
    image_collection_name: str,
    output_images_folder: str,
    images_path: List = None,
    images_dir: str = None,
) -> List:
    """Upload images from local path to gcs

    Args:
        image_collection_name (str): collection name
        output_images_folder (str): folder to store images in
        images_path (List, optional): path of images. Defaults to None.
        images_dir (str, optional): directory of images. Defaults to None.

    Raises:
        Exception: GCS Upload failed

    Returns:
        List: gcs path images were uploaded to
    """
    uploaded_images = []

    if images_dir:
        images_path = [
            os.path.abspath(os.path.join(images_dir, p))
            for p in os.listdir(images_dir)
            if p.endswith(("jpg", "png"))
        ]
    for image_path in images_path:
        filename = os.path.basename(image_path)
        gcs_path = os.path.join(
            "gs://voxel-logs",
            output_images_folder,
            image_collection_name,
            filename,
        )
        try:
            upload_to_gcs(gcs_path, image_path)
            uploaded_images.append(gcs_path)
        except Exception as gcs_upload_exception:
            logger.error(f"Couldnt upload to gcs {gcs_path}")
            raise gcs_upload_exception
    return uploaded_images


def copy_images(
    image_collection_name: str,
    output_images_folder: str,
    images_path: List = None,
    images_dir: str = None,
) -> List:
    """Copy images from gcs to voxel-logs

    Args:
        image_collection_name (str): collection name
        output_images_folder (str): folder to store images in
        images_path (List, optional): path of images. Defaults to None.
        images_dir (str, optional): directory of images. Defaults to None.

    Raises:
        Exception: Copy to gcs failed

    Returns:
        List: gcs path images were uploaded to
    """
    uploaded_images = []
    if images_dir:
        bucket, relative_gcs_path = separate_bucket_from_relative_path(
            images_dir
        )
        images_path = [
            os.path.join(f"gs://{bucket}", gcs_file.name)
            for gcs_file in get_files_in_bucket(
                bucket, prefix=relative_gcs_path
            )
            if gcs_file.endswith(("jpg", "png"))
        ]
    for image_path in images_path:
        _, relative_gcs_path = separate_bucket_from_relative_path(image_path)
        filename = os.path.basename(relative_gcs_path)
        gcs_path = os.path.join(
            "gs://voxel-logs",
            output_images_folder,
            image_collection_name,
            filename,
        )
        try:
            copy_blob(image_path, gcs_path)
            uploaded_images.append(gcs_path)
        except Exception as copy_gcs_exception:
            logger.error(f"Couldnt copy to gcs {gcs_path}")
            raise copy_gcs_exception
    return uploaded_images


def upload_images_to_log_store(
    output_images_folder: str,
    src: ImagesSource,
    images_path: List,
    images_dir: str,
    image_collection_name: str,
) -> None:
    """Upload images to log store from gcs, s3 or local path

    Args:
        output_images_folder (str): folder to store in
        src (ImagesSource): source of the images
        images_path (List, optional): path of the images. Defaults to None.
        images_dir (str, optional): directory of the image. Defaults to None.
        image_collection_name (str, optional): image collection name. Defaults to None.

    Raises:
        RuntimeError: Source of images is invalid
    """
    if src == ImagesSource.S3:
        copy_s3_images_to_voxel_logs(
            image_collection_name,
            output_images_folder,
            images_path,
            images_dir,
        )
    elif src == ImagesSource.LOCAL:
        upload_local_images_gcs(
            image_collection_name,
            output_images_folder,
            images_path,
            images_dir,
        )
    elif src == ImagesSource.GCS:
        copy_images(
            image_collection_name,
            output_images_folder,
            images_path,
            images_dir,
        )
    else:
        raise RuntimeError(f"Images src not of right now, specified {src}")

    logger.info(
        (
            f"Data collection uuid "
            f"{os.path.join(output_images_folder, image_collection_name)}"
        )
    )


def parse_args() -> object:
    """Parse arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    input_type = parser.add_mutually_exclusive_group()
    input_type.add_argument(
        "--images_path",
        metavar="I",
        type=str,
        nargs="+",
        help="List of images to ingest",
    )
    input_type.add_argument(
        "--images_dir",
        metavar="d",
        type=str,
        help="Dir with images",
    )
    folder_struct = parser.add_mutually_exclusive_group()
    folder_struct.add_argument(
        "--camera_uuid",
        type=str,
        help="camera uuid to upload images to",
    )
    parser.add_argument(
        "-t",
        "--is_test",
        type=str,
        help="Should data collection be ingested for test or not",
        default="false",
    )
    folder_struct.add_argument(
        "--synthetic_folder",
        type=str,
        help="folder to upload images to for synthetic data",
    )
    parser.add_argument(
        "--image_collection_name",
        type=str,
        default=str(uuid.uuid4()),
        help="Uuid to upload images to",
    )
    parser.add_argument(
        "-t",
        "--is_test",
        type=str,
        help="Should data collections be ingested for test or not",
        default="false",
    )
    parser.add_argument(
        "--src",
        type=str,
        default="GCS",
        help="location images are in",
        choices=[
            member.name
            for member in ImagesSource
            if member != ImagesSource.UNKNOWN
        ],
    )
    return parser.parse_args()


@sematic.func
def get_successfully_ingested_videos(
    successful_video_uuids: List[str],
) -> str:
    """Get video_uuid from a signle metaverse ingestion using a sematic future

    Args:
        successful_video_uuids (List[str]): successfully ingested video uuids
            returned by metaverse

    Returns:
        str: video uuid of successfully ingested video
    """
    return successful_video_uuids[0]


@sematic.func
def ingest_image_collection(
    output_folder: str,
    src: ImagesSource,
    images_path: List[str],
    image_collection_name: str,
    is_test: bool,
    images_dir: Optional[str] = None,
    metaverse_environment: Optional[str] = None,
) -> str:
    """Ingests image collection

    Args:
        output_folder (str): folder to store the images in
        src (ImagesSource): src of the images
        images_path (List): list of path of images
        images_dir (Optional[str]): directory of source of images
        image_collection_name (str): name of the image_collection
        is_test (bool): whether images are used for test
        metaverse_environment (Optional[str]): Metaverse environment for
            ingestion

    Returns:
        str: uuid of the data collection
    """
    image_collection_uuid = os.path.join(output_folder, image_collection_name)
    upload_images_to_log_store(
        output_folder,
        src,
        images_path,
        images_dir,
        image_collection_name,
    )
    logger.info("Ingesting data collections to metaverse")
    successfully_ingested_videos, _ = ingest_data_collections_to_metaverse(
        [
            DataCollectionInfo(
                data_collection_uuid=image_collection_uuid,
                is_test=is_test,
                data_collection_type=DataCollectionType.IMAGE_COLLECTION,
            )
        ],
        metaverse_environment=metaverse_environment,
    )
    return get_successfully_ingested_videos(successfully_ingested_videos)


if __name__ == "__main__":
    args = parse_args()
    if args.camera_uuid:
        img_collec_parent_fldr = args.camera_uuid
    else:
        img_collec_parent_fldr = os.path.join(
            "synthetic", args.synthetic_folder
        )
    data_collection_uuid = f"{args.image_dir}/{args.image_collection_name}"
    ingest_image_collection(
        output_folder=img_collec_parent_fldr,
        src=ImagesSource[args.src],
        images_path=args.images_path,
        images_dir=args.images_dir,
        image_collection_name=args.image_collection_name,
        is_test=args.is_test.upper() == "TRUE",
    ).resolve(SilentResolver())
