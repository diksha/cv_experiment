#
# Copyright 2020-2022 Voxel Labs, Inc.
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
import tempfile
from typing import List, Optional

import pandas as pd
import sematic
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.metaverse.metaverse import Metaverse
from core.structs.actor import ActorCategory
from core.utils.aws_utils import (
    separate_bucket_from_relative_path,
    upload_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bucket",
        metavar="B",
        type=str,
        default="voxel-datasets",
        help="Path to bucket to dump yolo dataset",
    )
    parser.add_argument(
        "-r",
        "--relative_path",
        metavar="R",
        type=str,
        required=True,
        help="Relative path to dump yolo dataset",
    )
    parser.add_argument(
        "-a",
        "--actors_to_keep",
        nargs="+",
        type=str,
        required=True,
    )

    return parser.parse_args()


class InvalidQueryException(Exception):
    """Exception for invalid graphql queries"""


def get_data_collection_metaverse_info(
    valid_video_uuids: list,
    metaverse_environment: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retrieve necessary info from metaverse for each voxel video uuid for training

    Args:
        valid_video_uuids (list): list of voxel_uuids for training
        metaverse_environment (Optional[str]): Metaverse environment to connect to

    Raises:
        InvalidQueryException: invalid graphql query

    Returns:
        metaverse_df (pd.DataFrame): dataframe containing necessary metaverse info
    """
    metaverse = Metaverse(environment=metaverse_environment)
    voxel_uuid = []
    missing_voxel_uuid = []
    uuid = []
    is_test = []
    for video_uuid in valid_video_uuids:
        query = """query get_data_collection_path_contains(
            $path: String!
        ) {
            data_collection_path_contains(path: $path) {
                uuid, is_test
            }
        }
        """
        qvars = {"path": video_uuid}
        result = metaverse.schema.execute(query, variables=qvars)
        if result.errors:
            raise InvalidQueryException(result.errors)
        data = result.data["data_collection_path_contains"]
        if data:
            voxel_uuid.append(video_uuid)
            uuid.append(data[0]["uuid"])
            is_test.append(data[0]["is_test"])
        else:
            missing_voxel_uuid.append(video_uuid)
    if missing_voxel_uuid:
        logger.warning(
            f"generate_yolo_dataset: {len(missing_voxel_uuid)} items not found!"
        )
    logger.info(f"generate_yolo_dataset: {len(voxel_uuid)} items found")

    video_info = {}
    video_info["voxel_uuid"] = voxel_uuid
    video_info["uuid"] = uuid
    video_info["is_test"] = is_test
    return pd.DataFrame(video_info)


def get_train_val_split(training_df: pd.DataFrame) -> tuple:
    """
    Splits data into train, and val

    Args:
        training_df (pd.DataFrame): dataframe of videos for training

    Returns:
        dataset_split (tuple): train and val dataframes
    """
    sites = training_df.voxel_uuid.str.split("/").str[0:2].str.join("/")
    unique_sites = list(sites.unique())
    train_split = []
    val_split = []
    for site in unique_sites:
        site_videos = training_df.voxel_uuid[
            training_df.voxel_uuid.str.contains(site)
        ].to_list()
        if len(site_videos) > 1:
            train_video, val_video = train_test_split(
                site_videos, test_size=0.2, random_state=42
            )
            train_split.extend(train_video)
            val_split.extend(val_video)
        else:
            logger.warning(
                f"Site {site} low on training videos, videos, {site_videos}"
            )
            train_split.extend(site_videos)
    train = training_df[training_df.voxel_uuid.isin(train_split)]
    val = training_df[training_df.voxel_uuid.isin(val_split)]
    return train, val


def generate_dataset_config(
    actor_names_list: list,
    nc: int,
    train_uuid_list: list,
    val_uuid_list: list,
    test_uuid_list: list,
    local_dir: str,
    upload_dir: str,
    dataset_name: str,
) -> str:
    """
    Generates dataset config used by yolo to train and test dataset

    Args:
        actor_names_list (list): names of actors in list format indexed by how
            dataset annotations were extracted
        nc (int): number of classes to train on
        train_uuid_list (list): train video uuid list (can be empty)
        val_uuid_list (list): val video uuid list (can be empty)
        test_uuid_list (list): test video uuid list (can be empty)
        local_dir (str): local path to save dataset
        upload_dir (str): gcs upload directory
        dataset_name (str): base name of yaml file to upload
    Returns:
        str: name of dataset YAML uploaded
    """
    upload_path = f"s3://{upload_dir}/{dataset_name}"
    (
        destination_bucket_name,
        destination_relative_s3_path,
    ) = separate_bucket_from_relative_path(upload_path)
    dataset_config = {
        "names": actor_names_list,
        "nc": nc,
        "train": [
            "/data/images/" + train_uuid for train_uuid in train_uuid_list
        ],
        "val": ["/data/images/" + val_uuid for val_uuid in val_uuid_list],
        "test": ["/data/images/" + test_uuid for test_uuid in test_uuid_list],
    }
    local_dataset_file_name = f"{local_dir}/{dataset_name}"
    with open(
        f"{local_dataset_file_name}", "w", encoding="UTF-8"
    ) as dataset_file:
        yaml.dump(dataset_config, dataset_file)
    upload_file(
        destination_bucket_name,
        local_dataset_file_name,
        destination_relative_s3_path,
    )
    return upload_path


def create_and_upload_datasets(
    dataset_dir: str,
    train_video_uuids: List[str],
    val_video_uuids: List[str],
    test_video_uuids: List[str],
    actors_to_keep: List[str],
) -> List[str]:
    """Create and upload all relevant dataset configurations for this task

    Args:
        dataset_dir (str): _description_
        train_video_uuids (List[str]): video UUIDs for training set
        val_video_uuids (List[str]): video UUIDs for validation set
        test_video_uuids (List[str]): video UUIDs for test set
        actors_to_keep (List[str]): names of actor types to include in data set

    Returns:
        List[str]: List of dataset YAML configurations created
    """
    actor_names = [actor.name for actor in actors_to_keep]
    train_dataset_path = f"{dataset_dir}/training"
    test_dataset_path = f"{dataset_dir}/testing"
    train_sites = {
        ("/").join(video_uuid.split("/")[0:2])
        for video_uuid in train_video_uuids
    }
    val_sites = {
        ("/").join(video_uuid.split("/")[0:2])
        for video_uuid in val_video_uuids
    }
    test_sites = {
        ("/").join(video_uuid.split("/")[0:2])
        for video_uuid in test_video_uuids
    }
    dataset_configs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset for training
        dataset_configs.append(
            generate_dataset_config(
                actor_names,
                len(actor_names),
                train_video_uuids,
                val_video_uuids,
                [],
                tmpdir,
                train_dataset_path,
                "yolo_training_dataset.yaml",
            )
        )
        # Create dataset for generalized testing
        dataset_configs.append(
            generate_dataset_config(
                actor_names,
                len(actor_names),
                [],
                [],
                test_video_uuids,
                tmpdir,
                test_dataset_path,
                "all_sites_testing_set.yaml",
            )
        )
        # Create dataset for train video site specific testing
        for site in train_sites:
            site_video_uuids = [
                video_uuid
                for video_uuid in train_video_uuids
                if site in video_uuid
            ]
            flattened_site = site.replace("/", "_")
            dataset_name = f"{flattened_site}_training_set.yaml"
            dataset_configs.append(
                generate_dataset_config(
                    actor_names,
                    len(actor_names),
                    [],
                    [],
                    site_video_uuids,
                    tmpdir,
                    test_dataset_path,
                    dataset_name,
                )
            )
        # Create dataset for val video site specific testing
        for site in val_sites:
            site_video_uuids = [
                video_uuid
                for video_uuid in val_video_uuids
                if site in video_uuid
            ]
            flattened_site = site.replace("/", "_")
            dataset_name = f"{flattened_site}_validation_set.yaml"
            dataset_configs.append(
                generate_dataset_config(
                    actor_names,
                    len(actor_names),
                    [],
                    [],
                    site_video_uuids,
                    tmpdir,
                    test_dataset_path,
                    dataset_name,
                )
            )
        # Create dataset for test video site specific testing
        for site in test_sites:
            site_video_uuids = [
                video_uuid
                for video_uuid in test_video_uuids
                if site in video_uuid
            ]
            flattened_site = site.replace("/", "_")
            dataset_name = f"{flattened_site}_testing_set.yaml"
            dataset_configs.append(
                generate_dataset_config(
                    actor_names,
                    len(actor_names),
                    [],
                    [],
                    site_video_uuids,
                    tmpdir,
                    test_dataset_path,
                    dataset_name,
                )
            )
    return dataset_configs


@sematic.func(resource_requirements=CPU_1CORE_4GB, standalone=True)
def generate_yolo_dataset(
    actors_to_keep: List[str],
    valid_uuids: List[str],
    bucket: str,
    relative_path: str,
    metaverse_environment: Optional[str] = None,
) -> List[str]:
    """Generate YOLO dataset

    Args:
        actors_to_keep (List[str]): list of actor types to keep
        valid_uuids (List[str]): list of valid UUIDs that have data and labels, for data sets
        bucket (str): cloud storage bucket to use for storing data
        relative_path (str): relative path to store data
        metaverse_environment (Optional[str]): Metaverse environment to connect to

    Returns:
        List[str]: list of dataset configuration paths created
    """
    actors_list = [ActorCategory[actor] for actor in actors_to_keep]
    video_df = get_data_collection_metaverse_info(
        valid_uuids, metaverse_environment
    )
    train_val_df = video_df[~video_df.is_test]
    test_df = video_df[video_df.is_test]
    train_df, val_df = get_train_val_split(
        train_val_df,
    )

    dataset_path = f"{bucket}/{relative_path}"
    return create_and_upload_datasets(
        dataset_path,
        train_df.voxel_uuid.to_list(),
        val_df.voxel_uuid.to_list(),
        test_df.voxel_uuid.to_list(),
        actors_list,
    )
