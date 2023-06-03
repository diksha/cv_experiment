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
import datetime
import json
import os

import cv2
import numpy as np
import torch.utils.data
from google.cloud import storage

from core.infra.cloud.gcs_utils import get_signing_credentials
from core.perception.inference.transforms import get_transforms
from core.structs.video import Video

from .transforms import ComposeTransform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, video_uuids, data_transforms=None):
        """
        num_of_historical_frames tells how many previous frame to add.
        """

        assert data_transforms is None or isinstance(
            data_transforms, list
        ), "Input data_transforms should be None or List"

        self.video_uuids = video_uuids

        if data_transforms:
            self.data_transforms = ComposeTransform(
                get_transforms(data_transforms)
            )
        else:
            self.data_transforms = data_transforms

        self.rows = self._get_rows()

    def __len__(self):
        return len(self.rows)

    def _get_rows(self):
        """
        Get the rows that needs to be fed into loader such as
        [ [video_uuid, [frame_idx1, frame_idx2 ....]] .... ]
        Basically only one frame idx per row if not multiple frame model
        but for tracking etc previous history needs to provided so create
        accordingly, use the num_of_historical_frames variable provided.
        """
        # TODO have a preprocessed dataset rather than reading frames from
        # video everytime.
        rows = []
        storage_client = storage.Client(project="sodium-carving-227300")
        for video_uuid in self.video_uuids:
            timestamps_frame_dict = {}
            label_path = os.path.join(
                os.environ["BUILD_WORKSPACE_DIRECTORY"],
                "data",
                "video_labels",
                "{}.json".format(video_uuid),
            )
            with open(label_path) as label_file:
                label_dict = json.load(label_file)
                gt_video = Video.from_dict(label_dict)
                for frame in gt_video.frames:
                    timestamps_frame_dict[
                        int(frame.relative_timestamp_ms)
                    ] = frame

            video_url = (
                storage_client.bucket("voxel-videos")
                .blob("{}.mp4".format(video_uuid))
                .generate_signed_url(
                    version="v4",
                    expiration=datetime.timedelta(minutes=30),
                    method="GET",
                    credentials=get_signing_credentials(),
                )
            )
            vcap = cv2.VideoCapture(video_url)
            count = 0
            while True:
                ret, frame = vcap.read()
                count += 1
                if not ret or count == 1000:
                    break
                if (
                    int(vcap.get(cv2.CAP_PROP_POS_MSEC))
                    in timestamps_frame_dict
                ):
                    rows.append(
                        (
                            frame,
                            timestamps_frame_dict[
                                int(vcap.get(cv2.CAP_PROP_POS_MSEC))
                            ],
                        )
                    )
        return rows

    def __getitem__(self, index):
        return self._load_sample(self.rows[index])

    def _load_sample(self, row):
        (frame, labels) = row
        converted_to_box_class = np.array(
            [actor.get_box() + [0] for actor in labels.actors]
        )
        return frame.transpose(2, 0, 1), converted_to_box_class

    def get_samples_for_inference(self, row):
        return self._load_sample(row)

    def params_to_log(self):
        return {"video_uuids": self.video_uuids, "num_samples": len(self.rows)}
