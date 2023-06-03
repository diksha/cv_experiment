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
import argparse
import json
import os
from tempfile import NamedTemporaryFile
from typing import Dict

import sematic
from lightly.api import ApiWorkflowClient
from loguru import logger
from sematic.resolvers.silent_resolver import SilentResolver

from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.ml.data.curation.voxel_lightly_utils import (
    LIGHTLY_TOKEN_ARN,
    create_and_configure_dataset,
)
from core.utils.aws_utils import (
    get_secret_from_aws_secret_manager,
    glob_from_bucket,
    upload_file,
)


def upload_filenames_to_s3(
    s3_filename: str, bucket: str, filenames: list, root: str
):
    """
    Uploads list of filenames to s3 given a s3 filename and a bucket

    Args:
        s3_filename (str): the s3 filename
        bucket (str): the bucket to output to
        filenames (list): the list of filenames
        root (str): root director of all the files

    Raises:
        ValueError: if the filenames are empty
    """
    if not filenames:
        raise ValueError(f"The filenames for {s3_filename} were empty")
    files = [os.path.relpath(file, root) for file in filenames]

    with NamedTemporaryFile() as tmp:
        with open(tmp.name, "w", encoding="utf-8") as output_file:
            output_file.writelines("\n".join(files))
        upload_file(
            bucket,
            tmp.name,
            s3_filename,
            extra_args={"ContentType": "text/plain"},
        )
        logger.info(f"Wrote files to: s3://{bucket}/{s3_filename}")


def create_dataset_and_save_id(
    camera_uuid: str, task: str, postfix: str = "", notify: bool = False
) -> Dict[str, str]:
    """Create a Lightly dataset and save the id to a file

    Args:
        camera_uuid (str): standard voxel camera_uuid
        task (str): the task to generate the dataset for
        postfix(str): postfix for directory used for datasets
        notify (bool): Whether to notify Slack or not on creation of dataset

    Returns:
        A dict with 4 values:
        - dataset_name: the name of the prepared dataset
        - dataset_id: the id of the prepared dataset
        - input_dir: the dir without the input bucket being written to
        - output_dir: the dir without the output bucket being written to
    """
    dataset_name = (
        f"{task}_{camera_uuid}{f'_{postfix}' if postfix else ''}".replace(
            "/", "_"
        )
    )
    input_dir = os.path.join(task, camera_uuid, postfix)
    output_dir = os.path.join(task, camera_uuid, postfix)
    lightly_client = ApiWorkflowClient(
        token=json.loads(
            get_secret_from_aws_secret_manager(LIGHTLY_TOKEN_ARN)
        )["1"]
    )
    dataset_id = create_and_configure_dataset(
        client=lightly_client,
        dataset_name=dataset_name,
        input_dir=input_dir,
        output_dir=output_dir,
        metadata={"camera_uuid": camera_uuid},
        notify=notify,
    )
    # save dataset id to file
    # trunk-ignore-all(bandit/B108)
    with open(
        "/tmp/dataset_id.txt", "w", encoding="utf-8"
    ) as dataset_id_file, open(
        "/tmp/lightly_token.txt", "w", encoding="utf-8"
    ) as token_file:
        dataset_id_file.write(dataset_id)
        token_file.write(
            json.loads(get_secret_from_aws_secret_manager(LIGHTLY_TOKEN_ARN))[
                "1"
            ]
        )
    return {
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "input_dir": input_dir,
        "output_dir": output_dir,
    }


def generate_relevant_files(
    input_bucket: str,
    output_bucket: str,
    project: str,
    camera_uuid: str,
    postfix: str,
    extension="mp4",
) -> None:
    """
    Generates the relevant_filenames.txt for the lightly run

    Args:
        input_bucket (str): the input bucket to look for the videos
        output_bucket (str): the output bucket to write the relevant filenames to
        project (str): the project name
        camera_uuid (str): the uuid for the run
        postfix (str): postfix for the directory to get relevant files from
        extension (str, optional): The extension for the videos. Defaults to "mp4".

    Raises:
        ValueError: raised when the input path is specified but there are not files
    """
    # generate relevant filenames
    root = os.path.join(project, camera_uuid, postfix)
    root = root if root.endswith("/") else f"{root}/"
    filenames = glob_from_bucket(input_bucket, root, (extension))
    logger.info(
        f"Found {len(filenames)} files matching s3://{input_bucket}/{root}*.{extension}."
    )
    if not filenames:
        raise ValueError(
            (
                f"The input bucket {input_bucket} and path: {root}"
                f"had no files with extension: *.{extension}"
            )
        )
    # create filenames
    all_s3_path = os.path.join(root, "all.txt")

    # upload
    upload_filenames_to_s3(all_s3_path, output_bucket, filenames, root)


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func(
    resource_requirements=CPU_1CORE_4GB,
    standalone=True,
)
def prepare_lightly_run(
    input_bucket: str,
    output_bucket: str,
    project: str,
    camera_uuid: str,
    postfix: str = "",
    extension: str = "mp4",
    notify: bool = False,
) -> Dict[str, str]:
    """# Prepares all the input files for the lightly run

    ## Parameters
    - **input_bucket**:
        The input bucket the Lightly execution will pull data from. Executing
        this function will write data to input_bucket to prepare it.
    - **output_bucket**:
        The output bucket the Lightly execution will write results to. Executing
        this function will write data to output_bucket to prepare it.
    - **project**:
        The output bucket the Lightly execution will write results to. Executing
        this function will write data to output_bucket to prepare it.
    - **camera_uuid**:
        The camera uuid for the job
    - **extension**:
        The extension for the videos. Defaults to "mp4".
    - **notify**:
        Whether to notify Slack or not on dataset creation. Defaults to False

    ## Returns
    A dict with 6 elements:
    - **dataset_name**: The name of the prepared dataset
    - **dataset_id**: The id of the prepared dataset
    - **input_bucket**: The name of the prepared input bucket
    - **output_bucket**: The name of the prepared output bucket
    - **input_dir**: The dir within the input bucket being read from
    - **output_dir**: The dir within the output bucket being written to
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    generate_relevant_files(
        input_bucket,
        output_bucket,
        project,
        camera_uuid,
        postfix,
        extension=extension,
    )

    result_dict = create_dataset_and_save_id(
        camera_uuid, project, postfix, notify=notify
    )
    result_dict["input_bucket"] = input_bucket
    result_dict["output_bucket"] = output_bucket
    return result_dict


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def get_preparation_field(
    lightly_prepatation_info: Dict[str, str], field_name: str
) -> str:
    """# Get the value of one of the prepared entities from prepare_lightly_run

    This serves not only to extract the fields, but also to express to Sematic
    a dependency on the prepared objects for downstream functions.

    ## Parameters
    - **lightly_prepatation_info**:
        The output from prepare_lightly_run
    - **field_name**:
        One of dataset_name, dataset_id, input_bucket, output_bucket, input_dir,
        output_dir

    ## Returns
    The value of the requested field.
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    # a descriptive KeyError will be generated if the field_name isn't
    # one of the expected values.
    return lightly_prepatation_info[field_name]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert camera config")
    parser.add_argument(
        "--camera-uuid",
        type=str,
        required=True,
        help="The camera uuid ",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=False,
        default="doors/cropped",
        help="Project to read from to generate relevant filenames",
    )
    parser.add_argument(
        "--output-bucket",
        type=str,
        required=False,
        default="voxel-lightly-output",
        help="The bucket to write the to.",
    )
    parser.add_argument(
        "--input-bucket",
        type=str,
        required=False,
        default="voxel-lightly-input",
        help="The bucket to read from.",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()
    prepare_lightly_run(
        args.input_bucket,
        args.output_bucket,
        args.project,
        args.camera_uuid,
    ).resolve(SilentResolver())
