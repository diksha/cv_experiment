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
Video UUIDs consist of two major components. Relative path of the video inside
the bucket containing voxel video logs and the name of the video. For example: a
video UUID can be folder1/folder2/video_name.mp4. The path folder1/folder2 represents
the path to the video inside the bucket that contains the video. For most videos it will
be voxel-logs bucket.
"""
import datetime
import os
import sys
from pathlib import Path
from typing import Optional

import google.auth
from google.api_core.exceptions import NotFound
from google.auth.transport import requests
from google.cloud import storage

from core.infra.cloud.utils import (
    get_service_account_credentials,
    use_service_account,
)

DEFAULT_PROJECT = "sodium-carving-227300"


def get_storage_client(project=DEFAULT_PROJECT):
    credentials = None
    if use_service_account():
        credentials = get_service_account_credentials(
            scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
        )
    elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is not None:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
        )
        auth_request = requests.Request()
        credentials.refresh(auth_request)
    return storage.Client(project=project, credentials=credentials)


def video_uuid_to_gcs_path(video_uuid):
    """
    If there are paths in videos uuid. For example, folder1/folder2/video_name
    then the gcs path is folder1/folder2. In case of video_uuid being "video_name"
    the returned path is "".

    Args:
        video_uuid (str): Video UUID.

    Returns:
        str: GCS path portion of provided video UUID.
    """
    path, _ = os.path.split(video_uuid)
    return path


def video_name_from_uuid(video_uuid):
    """
    See comments in video_uuid_to_gcs_path function.
    """
    _, name = os.path.split(video_uuid)
    return name.split(".")[0]


def get_video_uuid_from_gcs_path(gcs_path) -> Optional[str]:
    if not gcs_path:
        return None
    filename = get_filename_from_gcs_path(gcs_path)
    if filename:
        filename.split("_")[0]
    else:
        return None


def get_filename_from_gcs_path(gcs_path) -> Optional[str]:
    if not gcs_path:
        return None
    if len(gcs_path) < len("gs://"):
        return None
    return gcs_path[len("gs://") :].split("/")[-1]


def video_ext_from_uuid(video_uuid):
    _, name = os.path.split(video_uuid)
    name_components = name.split(".")
    if len(name_components) > 1:
        return name_components[1]
    else:
        """ """


def does_gcs_blob_exists(path, project=DEFAULT_PROJECT):
    if not is_valid_gcs_path_format(path):
        return False

    bucket_name, relative_gcs_path = separate_bucket_from_relative_path(path)
    client = get_storage_client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(relative_gcs_path)
    return blob.exists()


def is_valid_gcs_path_format(path):
    return path.startswith("gs://")


def separate_bucket_from_relative_path(path):
    if is_valid_gcs_path_format(path):
        path = path.replace("gs://", "")
        path_components = path.split("/")
        if len(path_components) > 1:
            return path_components[0], "/".join(path_components[1:])
        return path_components[0]
    return ""


def dump_mp4_to_gcs(
    relative_gcs_path,
    bucket,
    content,
    project=DEFAULT_PROJECT,
    storage_client=None,
):
    """
    Automatically assume the video format to be mp4. If this is incorrect, pass in explicit
    application type.

    Args:
        relative_gcs_path (str): Relative GCS path.
        bucket (str): GCS bucket name.
        content (bytes): Video data to dump to GCS.
        project (str): GCP project name.
        storage_client: Optional GCS storage client instance to use. If not
            provided, a new instance is created for each invocation. If you
            are calling this function within a loop, consider passing in a
            shared storage_client instance here.

    Returns:
        False if path is invalid, otherwise None.
    """
    if video_ext_from_uuid(relative_gcs_path) != "mp4":
        print(
            """Unsupported content type passed to dump_mp4_to_gcs. This function assumes content will only
         be of mp4 type"""
        )
        return False
    return dump_to_gcs(
        relative_gcs_path,
        bucket,
        content,
        "video/mp4",
        project,
        storage_client,
    )


def dump_to_gcs(
    path,
    content,
    content_type="application/json",
    project=DEFAULT_PROJECT,
    storage_client=None,
):
    """Dumps content to a GCS object.

    Given a valid bucket name and the path to a file inside that bucket, this function write a binary of the given
    content type to the provided path. If the path is not a valid gcs path, then the function simply returns False to indicate
    failure. If the write was successful, true is returned. By default all contents are considered to be json type. Pass in
    a valid content_type in the format, "video/mp4" etc. based on the type of object being sent.

    Args:
        path (str): GCS path to dump to.
        content (Union[str, bytes]): Object content.
        content_type (str): MIME type of the provided content.
        project (str): GCP project name.
        storage_client: Optional GCS storage client instance to use. If not
            provided, a new instance is created for each invocation. If you
            are calling this function within a loop, consider passing in a
            shared storage_client instance here.

    Returns:
        False if path is invalid, otherwise None.
    """
    bucket_name, relative_gcs_path = separate_bucket_from_relative_path(path)
    if bucket_name == "":
        return False
    if storage_client is None:
        storage_client = get_storage_client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(relative_gcs_path)
    return blob.upload_from_string(data=content, content_type=content_type)


def upload_to_gcs(
    gcs_path,
    local_file_path,
    content_type=None,
    project=DEFAULT_PROJECT,
    storage_client=None,
    metadata=None,
):
    bucket_name, relative_gcs_path = separate_bucket_from_relative_path(
        gcs_path
    )
    if bucket_name == "":
        return False
    if storage_client is None:
        storage_client = get_storage_client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(relative_gcs_path)
    if metadata:
        blob.metadata = metadata
    with open(local_file_path, "rb") as f:
        return blob.upload_from_file(f, content_type=content_type)


def get_files_in_bucket(
    bucket_name,
    project=DEFAULT_PROJECT,
    prefix=None,
    delimiter=None,
    limit=sys.maxsize,
    storage_client=None,
):
    if storage_client is None:
        storage_client = get_storage_client(project=project)
    bucket_iterator = storage_client.list_blobs(
        bucket_name, prefix=prefix, delimiter=delimiter
    )
    for resource in bucket_iterator:
        yield resource
        limit = limit - 1
        if limit <= 0:
            break


def read_from_gcs(
    path, project=DEFAULT_PROJECT, storage_client=None
) -> Optional[bytes]:
    bucket_name, relative_gcs_path = separate_bucket_from_relative_path(path)
    if bucket_name == "":
        return None
    if storage_client is None:
        storage_client = get_storage_client(project=project)
    bucket = storage_client.lookup_bucket(bucket_name)
    blob = bucket.get_blob(relative_gcs_path)
    if not blob:
        return None
    # NOTE: this returns bytes
    # https://googleapis.dev/python/storage/latest/blobs.html#google.cloud.storage.blob.Blob.download_as_string
    return blob.download_as_string()


def list_blobs_with_prefix(
    bucket_name,
    prefix,
    delimiter=None,
    project=DEFAULT_PROJECT,
    storage_client=None,
):
    """Lists all the blobs in the bucket that begin with the prefix.

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

    If you just specify prefix = 'a', you'll get back:
        a/1.txt
        a/b/2.txt

    However, if you specify prefix='a' and delimiter='/', you'll get back:
        a/1.txt

    Args:
        bucket_name (str): GCS bucket name.
        prefix (str): Prefix to search for.
        delimiter (str): Delimiter used with prefix to emulate hierarchy.
        project (str): GCP project name.
        storage_client: Optional GCS storage client instance to use. If not
            provided, a new instance is created for each invocation. If you
            are calling this function within a loop, consider passing in a
            shared storage_client instance here.

    Returns:
        list: List of matching blob names.
    """
    if storage_client is None:
        storage_client = get_storage_client(project=project)
    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefix, delimiter=delimiter
    )
    return [blob.name for blob in blobs]


def get_top_level_directories_for_path(
    bucket_name, prefix, project=DEFAULT_PROJECT, storage_client=None
):
    if storage_client is None:
        storage_client = get_storage_client(project=project)

    prefix = os.path.join(prefix, "/")
    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefix, delimiter="/"
    )
    next(blobs, ...)
    return blobs.prefixes


def blob_count_with_prefix(
    bucket_name,
    prefix,
    delimiter=None,
    project="sodium-carving-227300",
    storage_client=None,
):
    """See documentation of list_blobs_with_prefix"""
    if storage_client is None:
        storage_client = get_storage_client(project=project)
    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefix, delimiter=delimiter
    )
    return len(list(blobs))


def get_signed_url(
    bucket, filepath, project=DEFAULT_PROJECT, timeout_minutes=30, **config
):
    storage_client = get_storage_client(project=project)
    bucket = storage_client.bucket(bucket)
    if use_service_account():
        credentials = get_service_account_credentials(
            scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
        )
        return bucket.blob(filepath).generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=timeout_minutes),
            method="GET",
            credentials=credentials,
            **config,
        )
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is not None:
        return bucket.blob(filepath).generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=timeout_minutes),
            method="GET",
            credentials=None,
            **config,
        )
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
    )
    auth_request = requests.Request()
    credentials.refresh(auth_request)
    return bucket.blob(filepath).generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=timeout_minutes),
        method="GET",
        access_token=credentials.token,
        service_account_email="storageaccess@sodium-carving-227300.iam.gserviceaccount.com",
        **config,
    )


def set_blob_metadata(
    bucket_name, blob_name, metadata, project=DEFAULT_PROJECT
):
    storage_client = get_storage_client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.get_blob(blob_name)
    blob.metadata = metadata
    blob.patch()


def get_last_updated_timetamp(bucket_name, blob_name, project=DEFAULT_PROJECT):
    storage_client = get_storage_client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.get_blob(blob_name)
    return blob.updated


def get_blob_metadata(bucket_name, blob_name, project=DEFAULT_PROJECT):
    storage_client = get_storage_client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.get_blob(blob_name)
    return blob.metadata


def get_video_signed_url(
    video_uuid, bucket="voxel-logs", video_format="mp4"
) -> str:
    """Gets a signed url for the video uuid

    Args:
        video_uuid (str): video uuid to get signed url for
        bucket (str, optional): bucket video belongs to. Defaults to "voxel-logs".
        video_format (str, optional): format of video. Defaults to "mp4".

    Returns:
        str: signed url in string
    """
    relative_path = f"{video_uuid}.{video_format}"
    return get_signed_url(bucket, relative_path)


def download_files(
    path, output_path, project=DEFAULT_PROJECT, storage_client=None
) -> bool:
    bucket_name, relative_gcs_path = separate_bucket_from_relative_path(path)
    if bucket_name == "":
        return False
    if storage_client is None:
        storage_client = get_storage_client(project=project)
    bucket = storage_client.lookup_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=relative_gcs_path)  # Get list of files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        output_path_file = (
            output_path + "/" + blob.name.replace(relative_gcs_path, "")
        )
        directory = "/".join(output_path_file.split("/")[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(output_path_file)
    return True


def download_blob(
    path, output_path, project=DEFAULT_PROJECT, storage_client=None
):
    bucket_name, relative_gcs_path = separate_bucket_from_relative_path(path)
    if bucket_name == "":
        return None
    if storage_client is None:
        storage_client = get_storage_client(project=project)
    bucket = storage_client.lookup_bucket(bucket_name)
    blob = bucket.blob(relative_gcs_path)
    if not blob:
        return None
    return blob.download_to_filename(output_path)


def download_video_object_from_cloud(
    video_uuid,
    input_bucket,
    output_dir,
    input_prefix=None,
    video_format="mp4",
    storage_client=None,
):
    input_relative_path = f"{video_uuid}.{video_format}"
    if input_prefix:
        input_relative_path = os.path.join(input_prefix, input_relative_path)
    input_path = f"gs://{input_bucket}/{input_relative_path}"
    output_path = "{}/{}.{}".format(output_dir, video_uuid, video_format)
    folder_path = video_uuid.rsplit("/", 1)[0]
    os.makedirs("{}/{}".format(output_dir, folder_path), exist_ok=True)
    download_blob(input_path, output_path, storage_client)
    return output_path


def delete_blob(
    gcs_path, project=DEFAULT_PROJECT, storage_client=None, ignore_404=False
):
    if not gcs_path:
        return
    if not storage_client:
        storage_client = get_storage_client(project=project)
    bucket, path = separate_bucket_from_relative_path(gcs_path)
    blob = storage_client.bucket(bucket).blob(path)
    try:
        blob.delete()
    except NotFound as e:
        if not ignore_404:
            raise e


def batch_delete_blobs(*gcs_paths, project=DEFAULT_PROJECT, ignore_404=False):
    # Filter out None, empty string
    paths = [p for p in gcs_paths if p]
    storage_client = get_storage_client(project)
    try:
        with storage_client.batch():
            for path in paths:
                delete_blob(path, storage_client=storage_client)
    except NotFound as e:
        if not ignore_404:
            raise e


def copy_blob(
    source_uri: str,
    destination_uri: str,
    project: str = DEFAULT_PROJECT,
    storage_client: storage.Client = None,
):
    if not storage_client:
        storage_client = get_storage_client(project=project)
    source_bucket_name, source_path = separate_bucket_from_relative_path(
        source_uri
    )
    source_bucket = storage_client.get_bucket(source_bucket_name)
    source_blob = source_bucket.get_blob(source_path)

    (
        destination_bucket_name,
        destination_path,
    ) = separate_bucket_from_relative_path(destination_uri)
    destination_bucket = storage_client.get_bucket(destination_bucket_name)

    source_bucket.copy_blob(source_blob, destination_bucket, destination_path)
