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
import os
import tempfile
import typing
from dataclasses import dataclass

import cv2
import numpy as np
from sortedcontainers import SortedDict

from core.ml.data.generation.common.metadata import MetaData
from core.ml.data.generation.common.registry import StreamRegistry
from core.ml.data.generation.common.stream import Stream
from core.structs.data_collection import DataCollection, DataCollectionType
from core.utils.aws_utils import (
    download_to_file,
    separate_bucket_from_relative_path,
)
from core.utils.video_reader import S3VideoReader, S3VideoReaderInput


@dataclass
class DataCollectionMetaData(MetaData):
    source_name: str
    index: str
    time_ms: int

    def dump(self) -> typing.Tuple[str, str, str]:
        """
        Dumps the relavent video metadata out
        as a tuple

        Returns:
            typing.Tuple[str, str, str]: all the
                                   metadata listed as a tuple
        """
        return (self.source_name, self.index, self.time_ms)


@dataclass
class StreamData:
    """Data representing stream"""

    datum: np.ndarray
    label: typing.Any
    metadata: DataCollectionMetaData


class VideoFrameLabels:
    def __init__(self, labels: typing.Any, attribute: str, key: str):
        # example label stream
        self.synchronized_labels = SortedDict(
            {
                int(getattr(item, key)): item
                for item in getattr(labels, attribute)
            }
        )

    def index(self, key: int) -> typing.Any:
        """
        Indexes the frame labels at the particular key

        Args:
            key (int): the current key to check in the synchronized
                       labels

        Returns:
            typing.Any: the value of the label at the specified key
        """
        return self.synchronized_labels.get(key)


@StreamRegistry.register()
class SynchronizedVideoStream(Stream):
    def __init__(self, video):
        video_input = S3VideoReaderInput(video.name, "mp4", "voxel-logs")
        self.name = video.name
        self.video_reader = S3VideoReader(video_input)
        self.labels = VideoFrameLabels(
            video, "frames", "relative_timestamp_ms"
        )

    def stream(self) -> StreamData:
        """
        Streams the result of the video with the synchronized labels

        Yields:
            Iterator[typing.Tuple[np.ndarray, typing.Any, VideoMetadata]]: the
                            image, label and metadata tuple as an iterable
        """
        for frame_index, datum in enumerate(self.video_reader.read()):
            time = datum.relative_timestamp_ms
            label = self.labels.index(time)
            yield StreamData(
                datum.image,
                label,
                DataCollectionMetaData(
                    source_name=self.name, index=frame_index, time_ms=time
                ),
            )


@StreamRegistry.register()
class SynchronizedImageCollectionStream(Stream):
    def __init__(self, image_collection):
        self.image_collection = image_collection

    def stream(self) -> StreamData:
        """
        Streams the result of the video with the synchronized labels

        Yields:
            Iterator[typing.Tuple[np.ndarray, typing.Any, VideoMetadata]]: the
                            image, label and metadata tuple as an iterable
        """
        for frame in self.image_collection.frames:
            s3_path = os.path.join(
                os.path.splitext(self.image_collection.path)[0],
                frame.relative_image_path,
            )
            bucket, path = separate_bucket_from_relative_path(s3_path)
            with tempfile.NamedTemporaryFile() as temp:
                download_to_file(bucket, path, temp.name)
                image = cv2.imread(temp.name)
                yield StreamData(
                    image,
                    frame,
                    DataCollectionMetaData(
                        source_name=self.image_collection.name,
                        index=os.path.splitext(frame.relative_image_path)[0],
                        time_ms=0,
                    ),
                )


@StreamRegistry.register()
class DataCollectionStream(Stream):
    def __init__(self, data_collection: DataCollection):

        self.reader = None

        if (
            data_collection.data_collection_type
            == DataCollectionType.VIDEO.name
        ):
            self.reader = SynchronizedVideoStream(data_collection)
        elif (
            data_collection.data_collection_type
            == DataCollectionType.IMAGE_COLLECTION.name
        ):
            self.reader = SynchronizedImageCollectionStream(data_collection)
        else:
            raise ValueError(
                "Stream reader not found for data_collection_type:"
                f" {data_collection.data_collection_type}"
            )

    def stream(self) -> typing.Any:
        """
        Streams the results of the locally parsed reader if it
        exists. Please see SynchronizedVideoStream and the other
        readers for more information on their potential outputs

        Returns:
            (typing.Any): the result of the other reader streams
        """
        return self.reader.stream()
