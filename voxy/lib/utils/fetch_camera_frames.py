import base64
import itertools
from datetime import datetime
from typing import Any, Iterator

import boto3
import pytz

# trunk-ignore(pylint/E0611)
from protos.perception.structs.v1.frame_pb2 import CameraFrame

FRAME_STRUCTS_BUCKET = "voxel-perception-production-frame-structs"


def _pairwise(iterable: Iterator[Any]) -> Iterator[Any]:
    """Performs a pairwise iteration of "AB BC CD DE..."

    Args:
        iterable (Iterator): an iterable type

    Returns:
        Iterator: a pairwise iterator
    """
    it0, it1 = itertools.tee(iterable)
    next(it1, None)
    return zip(it0, it1)


def _datetime_from_key(key: str) -> datetime:
    """Produces a valid datetime object from an object name

    Args:
        key (str): object name in s3

    Returns:
        datetime: timezone aware datetime
    """
    # object names all look like this:
    # perception-frame-structs-1-2023-01-24-05-09-51-75427c0a-6f01-36f2-af69-440523eb6fe1
    obj_name = key.split("/")[-1]
    obj_timestr = "-".join(obj_name.split("-")[4:-5])
    fmt = "%Y-%m-%d-%H-%M-%S"
    return datetime.strptime(obj_timestr, fmt).replace(tzinfo=pytz.UTC)


def _fetch_camera_frame_object_keys(camera_uuid: str) -> Iterator[str]:
    """Returns an iterator that fetches frame struct object keys for a given camera uuid

    Args:
        camera_uuid (str): camera_uuid to look up

    Yields:
        Iterator[str]: iterator of object keys
    """
    client = boto3.client("s3")

    response = client.list_objects_v2(
        Bucket=FRAME_STRUCTS_BUCKET,
        Prefix=f"data/{camera_uuid}",
    )

    for obj in response["Contents"]:
        yield obj["Key"]

    while response["IsTruncated"]:
        response = client.list_objects_v2(
            Bucket=FRAME_STRUCTS_BUCKET,
            Prefix=f"data/{camera_uuid}",
            ContinuationToken=response["NextContinuationToken"],
        )

        for obj in response["Contents"]:
            yield obj["Key"]


# TODO: add parameters to control the time range
def fetch_camera_frames(
    camera_uuid: str,
    start: datetime = datetime.min,
    end: datetime = datetime.max,
) -> Iterator[CameraFrame]:
    """Fetches camera frame structs from s3 with optional coarse time-based filtering.

    Args:
        camera_uuid (str): a valid camera uuid
        start (datetime, optional): an optional start time, up to 15 minutes
                                    of content before this point may be delivered
        end (datetime, optional): an optional end time, up to 15 minutes of content
                                  after this point may be delivered

    Yields:
        CameraFrame: camera frame containing an embedd frame struct

    Raises:
        RuntimeError: invalid input or no content found
    """
    client = boto3.client("s3")

    if end <= start:
        raise RuntimeError("invalid end before start specified")

    for key0, key1 in _pairwise(_fetch_camera_frame_object_keys(camera_uuid)):
        # if our end marker is before the next key, break as we are done
        if end < _datetime_from_key(key0):
            break

        # if n+1 is None this is our last element so we always send it as
        # there is no way to determine whether the start of our data might
        # be inside this element.

        # if our start marker is before the key at n+1
        # then we send the object at key n since we know
        # our data starts somewhere between the start of n and n+1
        if key1 is None or start < _datetime_from_key(key1):
            response = client.get_object(
                Bucket=FRAME_STRUCTS_BUCKET,
                Key=key0,
            )
            for line in response["Body"].iter_lines():
                yield CameraFrame.FromString(base64.b64decode(line))
