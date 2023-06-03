#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import base64
import glob
import io
import json
import os
import pathlib
import sys
import tempfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
from botocore.exceptions import ClientError
from loguru import logger
from mypy_boto3_secretsmanager import Client as SecretsManagerClient
from mypy_boto3_secretsmanager.type_defs import (
    FilterTypeDef,
    SecretListEntryTypeDef,
)
from PIL import Image

from core.infra.cloud.gcs_utils import download_blob


def get_blobs_from_bucket(
    bucket: str,
    prefix: str,
    delimiter: str = "",
    limit: int = sys.maxsize,
):
    """Get blobs from bucket
    Args:
        bucket (str): bucket to pull object from
        prefix (str): directory in the bucket to download
        delimiter (str): optional delimiter for filtering
        limit (int): max limit of files to download
    Yields:
        s3.ObjectSummary: blob for resource from s3
    """
    s3_resource = boto3.resource("s3")
    for resource in s3_resource.Bucket(bucket).objects.filter(
        Prefix=prefix, Delimiter=delimiter
    ):
        yield resource

        limit = limit - 1
        if limit <= 0:
            break


def get_blobs_contents_from_bucket(
    bucket: str,
    prefix: str,
    delimiter: str = "",
    limit: int = sys.maxsize,
):
    """Get blobs from bucket
    Args:
        bucket (str): bucket to pull object from
        prefix (str): directory in the bucket to download
        delimiter (str): optional delimiter for filtering
        limit (int): max limit of files to download
    Yields:
        s3.ObjectSummary: blob for resource from s3
    """
    s3_resource = boto3.resource("s3")
    for resource in s3_resource.Bucket(bucket).objects.filter(
        Prefix=prefix, Delimiter=delimiter
    ):
        yield resource.get()["Body"].read()

        limit = limit - 1
        if limit <= 0:
            break


def get_bucket_path_from_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Get bucket name and path from s3_uri

    Args:
        s3_uri (str): uri of s3 file

    Returns:
        Tuple[str,str]: bucket name and path
    """
    parse = urlparse(s3_uri, allow_fragments=False)
    return parse.netloc, parse.path.lstrip("/")


def download_video_object_from_cloud(
    video_uuid: str,
    input_bucket: str,
    output_dir: str,
    input_prefix: Optional[str] = None,
    video_format: str = "mp4",
) -> str:
    """Download a video from s3

    The downloaded URI is constructed as:
    s3://{input_bucket}/[{input_prefix}/?]{video_uuid}.{video_format}

    The video is downloaded to:
    {output_dir}/{video_uuid}.{video_format}

    Args:
        video_uuid: The UUID of the video to download
        input_bucket: The s3 bucket to download the video from
        output_dir: The local directory to download the video into
        input_prefix: The prefix to put between the bucket and the video_uuid to get the file
        video_format: The extension for the video format (defaults to "mp4")

    Returns:
        The path that the video is downloaded to

    Raises:
        RuntimeError: If the video could not be downloaded
    """
    file_key = f"{video_uuid}.{video_format}"
    if input_prefix:
        source_key = os.path.join(input_prefix, file_key)
    else:
        source_key = file_key
    s3_client = boto3.resource("s3")
    output_path = f"{output_dir}/{video_uuid}.{video_format}"
    folder_path = video_uuid.rsplit("/", 1)[0]
    os.makedirs(f"{output_dir}/{folder_path}", exist_ok=True)
    try:
        s3_client.Bucket(input_bucket).download_file(source_key, output_path)
    except ClientError as exc:
        raise RuntimeError(
            f"Error downloading s3://{input_bucket}/{source_key}: {exc}"
        ) from exc
    return output_path


def download_to_file(bucket: str, s3_path: str, local_path: str) -> str:
    """
    Downloads from a s3 bucket to a file

    Args:
        bucket (str): the bucket to grab from
        s3_path (str): the s3 path in the bucket to download
        local_path (str): the local file path to download to

    Returns:
        str: the local file path for the downloaded file

    Raises:
        RuntimeError: if there is an error downloading the file
    """
    s3_client = boto3.client("s3")
    try:
        s3_client.download_file(bucket, s3_path, local_path)
    except ClientError as exc:
        raise RuntimeError(
            f"Error downloading s3://{bucket}/{s3_path}: {exc}"
        ) from exc
    return local_path


def glob_from_bucket(bucket: str, prefix: str, extensions: tuple) -> list:
    """
    Globs all files with prefix path and a given extension

    Args:
        bucket (str): the bucket to glob from
        prefix (str): the prefix for the s3 query
        extensions (tuple): tuple of extensions to grab from

    Returns:
        list: the list of paths inside the bucket with the given prefix and extension
    """
    s3_client = boto3.client("s3")

    response = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
    if "Contents" not in response or not response["Contents"]:
        return []
    responses = response["Contents"]
    while response["IsTruncated"]:
        # From: https://boto3.amazonaws.com/v1/documentation/api/
        # latest/reference/services/s3.html#S3.Client.list_objects
        # If response does not include the NextMarker and it is truncated,
        # you can use the value of the last Key in the response as the marker
        # in the subsequent request to get the next set of object keys.
        next_marker = response["Contents"][-1]["Key"]
        response = s3_client.list_objects(
            Bucket=bucket, Prefix=prefix, Marker=next_marker
        )
        if "Contents" in response:
            responses.extend(response["Contents"])
        else:
            logger.warning(
                "Response was invalid for next marker, exiting early"
            )
            break

    filenames = [item["Key"] for item in responses]
    globbed_filenames = [
        filename for filename in filenames if filename.endswith(extensions)
    ]
    return globbed_filenames


def download_directory_from_s3(
    local_path: str, bucket_name: str, prefix: str, processes: int = 10
):
    """Downloads the s3 directory locally

    Thread safety based on boto3 documentation:
        https://boto3.amazonaws.com/v1/documentation/..
            api/latest/guide/clients.html#multithreading-or-multiprocessing-with-clients

    Args:
        local_path (str): Path to download files to
        bucket_name (str): name of s3 bucket
        prefix (str): directory with ending / in the bucket to download.
        processes (int): the number of processes to thread the download. Defaults to 10.
    """
    if prefix[-1] != "/":
        prefix = prefix + "/"
    s3_client = boto3.client("s3")

    def download_object(key: str):
        target = (
            key
            if local_path is None
            else local_path / Path(key).relative_to(prefix)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if key[-1] == "/":
            return
        s3_client.download_file(bucket_name, key, str(target))

    filenames = glob_from_bucket(
        bucket=bucket_name, prefix=prefix, extensions=""
    )
    pool = ThreadPool(processes=processes)
    pool.map(download_object, filenames)


def upload_file(
    bucket: str,
    local_path: str,
    s3_path: str,
    extra_args: Optional[dict] = None,
) -> str:
    """
    Uploads file to S3

    Args:
        bucket (str): the bucket to upload to
        local_path (str): the local file path
        s3_path (str): the destination path in the bucket
        extra_args (Optional[dict]): extra arguments to pass to S3 upload

    Returns:
        str: the destination S3 path
    """
    s3_client = boto3.client("s3")
    s3_client.upload_file(
        local_path,
        bucket,
        s3_path,
        ExtraArgs=extra_args if extra_args is not None else {},
    )
    return s3_path


def upload_directory_to_s3(
    bucket: str, local_directory: str, s3_directory: str, processes: int = 10
) -> str:
    """
    Uploads local directory to S3 bucket. Thread safety based on boto documentation:
        https://boto3.amazonaws.com/v1/documentation/..
            api/latest/guide/clients.html#multithreading-or-multiprocessing-with-clients

    Args:
        bucket (str): the bucket to upload to
        local_directory (str): the local directory to be syncing from
        s3_directory (str): the destination s3 directory to copy to
        processes (int, optional): the number of processes to upload simultaneously. Defaults to 10.

    Returns:
        str: the S3 directory path
    """
    s3_client = boto3.client("s3")
    glob_query = os.path.join(local_directory, "**")
    filenames = glob.glob(glob_query, recursive=True)

    def upload_single_file(current_file: str) -> None:
        """
        Uploads single file to S3

        Args:
            current_file (str): the current local file to upload
        """
        if os.path.isdir(current_file):
            return
        s3_filename = f"{current_file.replace(local_directory, '')}"
        s3_filename = os.path.join(s3_directory, s3_filename.lstrip("/"))
        s3_client.upload_file(current_file, bucket, s3_filename)

    pool = ThreadPool(processes=processes)
    pool.map(upload_single_file, filenames)
    return f"s3://{bucket}/{s3_directory}"


def get_secret_manager_client() -> SecretsManagerClient:
    """Get secret manager client

    Returns:
        SecretsManagerClient: secret manage client
    """
    region_name = "us-west-2"
    session = boto3.session.Session()
    return session.client(
        service_name="secretsmanager", region_name=region_name
    )


def list_secrets(filters: List[FilterTypeDef]) -> List[SecretListEntryTypeDef]:
    """Get list of secrets

    Args:
        filters (List[FilterTypeDef]): filters to get the list

    Returns:
        List[SecretListEntryTypeDef]: secret list
    """
    client = get_secret_manager_client()
    response = client.list_secrets(Filters=filters)
    return response["SecretList"]


def get_secret_from_aws_secret_manager(secret_id: str) -> Union[bytes, str]:
    """Get secret from aws secret manager

    Args:
        secret_id (str): secret id

    Returns:
        Union[bytes, str]: values from aws secret manager
    """
    return get_value_from_aws_secret_manager(secret_id, "")


def get_value_from_aws_secret_manager(
    secret_id: str, key: str
) -> Union[bytes, str]:
    """Get values from aws secret manager

    Args:
        secret_id (str): secret id
        key (str): key

    Raises:
        err: error code is DecryptionFailureException
        err: error code is InternalServiceErrorException
        err: error code is InvalidParameterException
        err: error code is InvalidRequestException
        err: error code is DecryptionFailureException
        err: client error

    Returns:
        Union[bytes, str]: values
    """
    client = get_secret_manager_client()

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.
    if secret_id.upper() in os.environ:
        return os.environ[secret_id.upper()]

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_id)
    except ClientError as err:
        if err.response["Error"]["Code"] == "DecryptionFailureException":
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise err
        if err.response["Error"]["Code"] == "InternalServiceErrorException":
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise err
        if err.response["Error"]["Code"] == "InvalidParameterException":
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise err
        if err.response["Error"]["Code"] == "InvalidRequestException":
            # You provided a parameter value that is not valid
            # for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise err
        if err.response["Error"]["Code"] == "ResourceNotFoundException":
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise err
        raise err

    # Decrypts secret using the associated KMS CMK.
    # Depending on whether the secret is a string or binary, one of these fields will be populated.
    if "SecretString" in get_secret_value_response:
        secret_string = get_secret_value_response["SecretString"]
        looks_like_json = (
            len(secret_string) > 0
            and secret_string.startswith("{")
            and secret_string.endswith("}")
        )
        if looks_like_json and key:
            secret_dict = json.loads(secret_string)
            return secret_dict[key]
        return secret_string

    # Value is binary
    decoded_binary_secret = base64.b64decode(
        get_secret_value_response["SecretBinary"]
    )
    return decoded_binary_secret


def create_aws_secret(name: str, secret_value: Union[str, bytes]) -> None:
    """Create a secret with the given name and set value to secret_value.

    Args:
        name (str): Unique identifier for secrete in AWS secrets manager
        secret_value (Union[str, bytes]): The value to be stored in the secrets maanager

    Returns:
        None
    """
    client = get_secret_manager_client()
    try:
        kwargs = {"Name": name}
        if isinstance(secret_value, str):
            kwargs["SecretString"] = secret_value
        elif isinstance(secret_value, bytes):
            kwargs["SecretBinary"] = secret_value

            response = client.create_secret(**kwargs)
            logger.info(f"Created secret {name} in aws secrets manager")
    except ClientError as exc:
        logger.exception(f"Creation of secret failed {exc}")
        raise exc
    return response


def update_aws_secret(secret_id: str, secret_value: Union[str, bytes]) -> None:
    """Update an existing secret with the given name.

    Args:
        secret_id (str): Unique identifier for secret in AWS secrets manager
        secret_value (Union[str, bytes]): The value to be stored in the secrets maanager

    Returns:
        None
    """
    client = get_secret_manager_client()
    try:
        kwargs = {"SecretId": secret_id}
        if isinstance(secret_value, str):
            kwargs["SecretString"] = secret_value
        elif isinstance(secret_value, bytes):
            kwargs["SecretBinary"] = secret_value

        response = client.update_secret(**kwargs)
        logger.info(f"Updated secret {secret_id} in aws secrets manager")
    except ClientError as exc:
        logger.exception(f"Update of secret failed {exc}")
        raise exc
    return response


def is_valid_s3_path_format(path: str) -> bool:
    """Checks if s3 path is in a valid format
    Args:
        path (str): path of the s3 URI
    Returns:
        bool: true or false if is valid s3 path format
    """
    return path.startswith("s3://")


def does_s3_blob_exists(path):
    """Checks if a s3 object exists at the given path
    Args:
        path (str): path of the s3 URI
    Returns:
        bool: true or false if a s3 object exists at the given path.
              Returns false if the path is not a well formatted s3 path or if the file doesnt exist.
    """
    if not is_valid_s3_path_format(path):
        return False

    bucket, prefix = separate_bucket_from_relative_path(path)
    s3_client = boto3.client("s3")

    response = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
    if (
        "Contents" not in response
        or not response["Contents"]
        or len(response["Contents"]) != 1
    ):
        return False

    return True


def list_blobs_with_prefix(bucket: str, prefix: str):
    """Returns some or all (up to 1,000) of the objects in a bucket that start with a given prefix.

    Args:
        bucket (str): bucket to look for objects
        prefix (str): prefix to filter objects by

    Returns:
        dict: Metadata about each object returned.
    """
    s3_client = boto3.client("s3")

    response = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
    return response["Contents"]


def separate_bucket_from_relative_path(
    path: str,
) -> Union[str, Tuple[str, str]]:
    """Separates bucket from relative path
    Args:
        path (str): full s3 bucket path
    Returns:
        Union[str, Tuple[str, str]]: either a single string or both the bucket and the path
    """
    if is_valid_s3_path_format(path):
        path = path.replace("s3://", "")
        path_components = path.split("/")
        if len(path_components) > 1:
            return path_components[0], "/".join(path_components[1:])
        return path_components[0]
    return ""


def generate_presigned_url(
    bucket: str,
    filepath: str,
    timeout: int = 300,
    response_disposition: Optional[str] = None,
    s3_client: boto3.Session.client = None,
) -> Union[str, None]:
    """Generates a presigned URL from s3
    Args:
        bucket (str): name of the bucket
        filepath (str): filepath of the bucket
        timeout (int, optional): ttl of the signed url in seconds. Defaults to 120.
        response_disposition (Optional[str], optional): content resp disposition. Defaults to None.
        s3_client (boto3.Session.client): A boto3 client instance.
    Returns:
        Union[str, None]: returns either none or a presigned url
    """
    params = {
        "Bucket": bucket,
        "Key": filepath,
    }

    if response_disposition:
        params["ResponseContentDisposition"] = response_disposition

    if not s3_client:
        s3_client = boto3.client("s3")

    try:
        return s3_client.generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=timeout,
        )
    except ClientError as err:
        logger.error(err)
        return None


def batch_delete_s3_files(*s3_paths, ignore_404=False):
    """Batch deletes s3 files from s3 paths
    Args:
        *s3_paths: array of potential s3 paths
        ignore_404 (bool, optional): ignore client errors or not. Defaults to False.
    Raises:
        ClientError: raises not found error
    """
    batch_delete = {}

    for path in s3_paths:
        (
            bucket_name,
            relative_s3_path,
        ) = separate_bucket_from_relative_path(path)

        if batch_delete.get(bucket_name):
            batch_delete[bucket_name].append(
                {
                    "Key": relative_s3_path,
                }
            )
        else:
            batch_delete[bucket_name] = [
                {
                    "Key": relative_s3_path,
                }
            ]

    s3_client = boto3.client("s3")
    for bucket, objects in batch_delete.items():
        try:
            s3_client.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": objects,
                },
            )
        except ClientError as err:
            if not ignore_404:
                raise err


def does_blob_exist(path: str) -> bool:
    """Check to see if file exists
    Args:
        path (str): the path of the s3 file
    Returns:
        bool: does the file exist
    """
    s3_resource = boto3.resource("s3")
    (
        bucket_name,
        relative_s3_path,
    ) = separate_bucket_from_relative_path(path)
    if bucket_name == "":
        return False
    blob = list(
        s3_resource.Bucket(bucket_name).objects.filter(Prefix=relative_s3_path)
    )
    return len(blob) == 1 and blob[0].key == relative_s3_path


def read_from_s3(path: str) -> Optional[bytes]:
    """Reads from s3 bucket
    Args:
        path (str): the path of the s3 bucket
    Returns:
        Optional[bytes]: the json file contained in the path
    """
    (
        bucket_name,
        relative_s3_path,
    ) = separate_bucket_from_relative_path(path)
    if bucket_name == "":
        return None

    bytes_buffer = io.BytesIO()

    s3_client = boto3.client("s3")
    s3_client.download_fileobj(
        Bucket=bucket_name, Key=relative_s3_path, Fileobj=bytes_buffer
    )
    return bytes_buffer.getvalue()


def read_decoded_bytes_from_s3(path: str) -> Optional[str]:
    """Reads from s3 bucket
    Args:
        path (str): the path of the s3 bucket
    Returns:
        Optional[str]: the decoded json file contained in the path
    """
    file_bytes = read_from_s3(path)
    if file_bytes:
        file_bytes = file_bytes.decode("utf-8")
    return file_bytes


def upload_fileobj_to_s3(
    path: str, content: bytes, content_type: str = "application/json"
) -> bool:
    """Upload file to s3
    Args:
        path (str): path of the fileobj that we need to upload to s3
        content (bytes): content that we are uploading to s3
        content_type (str): content type of data. Default "application/json".
    Returns:
        bool: successful upload boolean
    """
    (
        bucket_name,
        relative_s3_path,
    ) = separate_bucket_from_relative_path(path)
    if bucket_name == "":
        return False

    s3_client = boto3.client("s3")
    s3_client.upload_fileobj(
        io.BytesIO(content),
        bucket_name,
        relative_s3_path,
        {
            "ContentType": content_type,
        },
    )
    return True


def download_cv2_imageobj_to_memory(bucket_name: str, path: str) -> np.array:
    """Download an image from s3 to memory

    Args:
        bucket_name (str): name of the bucket
        path (str): path to the image

    Returns:
        np.array: image in memory
    """
    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket_name, Key=path)
    image_data = response["Body"].read()

    # Convert image data to ndarray
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)


def upload_cv2_imageobj_to_s3(path: str, image: np.array) -> bool:
    """
    Wrapper for uploading images using upload_fileobj_to_s3
    Args:
        path (str): s3 file path to upload image
        image (np.array): image array to upload to s3
    Returns:
        bool: upload success
    """
    img_extension = pathlib.Path(path).suffix
    with tempfile.NamedTemporaryFile(suffix=img_extension) as temp_file:
        cv2.imwrite(temp_file.name, image)
        (
            bucket_name,
            relative_s3_path,
        ) = separate_bucket_from_relative_path(path)
        upload_file(
            bucket_name,
            temp_file.name,
            relative_s3_path,
        )
        return True


def copy_object(source_uri: str, destination_uri: str) -> None:
    """Creates a copy of an object already stored in S3

    Args:
        source_uri (str): full path of the source object
        destination_uri (str): full path of the destination object
    """
    (
        destination_bucket_name,
        destination_relative_s3_path,
    ) = get_bucket_path_from_s3_uri(destination_uri)
    if destination_bucket_name == "":
        return

    (
        source_bucket_name,
        source_relative_s3_path,
    ) = get_bucket_path_from_s3_uri(source_uri)

    s3_client = boto3.client("s3")
    s3_client.copy_object(
        Bucket=destination_bucket_name,
        Key=destination_relative_s3_path,
        CopySource={
            "Bucket": source_bucket_name,
            "Key": source_relative_s3_path,
        },
    )


def copy_gcs_to_aws(gcs_path: str, s3_uri: str):
    """Copy gcs file to aws

    Args:
        gcs_path (str): gcs full formed path of format gs://bucket/path
        s3_uri (str): s3 uri of destination path to copy to
    """
    bucket, s3_path = get_bucket_path_from_s3_uri(s3_uri)
    with tempfile.NamedTemporaryFile() as temp:
        download_blob(gcs_path, temp.name)
        upload_file(bucket, temp.name, s3_path)
