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

# "activity": "{"activity": "UNKNOWN", "posture": "BAD"}",

import json

import cv2

from core.infra.cloud.gcs_cv2_utils import upload_cv2_image_to_gcs
from core.infra.cloud.gcs_utils import get_storage_client, get_video_signed_url
from core.labeling.label_store.label_reader import LabelReader
from core.metaverse.metaverse import Metaverse
from core.perception.pose.api import PoseModel
from core.perception.pose.pose_embedder import KeypointPoseEmbedder
from core.structs.actor import ActorCategory, OccludedDegree, OperatingObject
from core.structs.attributes import RectangleXYWH
from core.structs.ergonomics import ActivityType, PostureType
from core.structs.video import Video


# trunk-ignore-all(pylint/R1702,pylint/R0914)
class DatasetGeneratorPersonPoseClassification:
    def __init__(
        self,
        output_path,
        data_source=None,
        activity_type=ActivityType.LIFTING,
    ):
        self.output_path = output_path
        self.label_reader = LabelReader()
        self.activity_type = activity_type
        self.storage_client = get_storage_client()
        self.data_source = data_source
        self.activity_dict = {
            ActivityType.REACHING: "reach",
            ActivityType.LIFTING: "lift",
        }
        self._pose_model = PoseModel(
            "artifacts_03_21_2023_pose_0630_jit_update/fast_res50_256x192.pt"
        )

    def generate_for_video_uuid(self, video_reference: dict):
        """Generate labels and images for a video uuid

        Args:
            video_reference (dict): Dictionary of the reference of the video.
        """
        video_uuid = json.loads(video_reference)["video"][0]["uuid"]
        query = """
            query video($uuid: String) {
                video(uuid: $uuid) {
                    uuid, name, path, frame_ref {
                        frame_number,  actors_ref {
                            polygon, category, occluded, operating_object, operating_pit, occluded_degree, truncated, activity
                        }
                    }
                }
            }
        """
        qvars = {"uuid": video_uuid}
        result = Metaverse(environment="PROD").schema.execute(
            query, variables=qvars
        )

        for video in result.data["video"]:
            vcap = (
                cv2.VideoCapture(
                    get_video_signed_url(
                        video_uuid, bucket="voxel-datasets", video_format="png"
                    )
                )
                if self.data_source == "original"
                else cv2.VideoCapture(get_video_signed_url(video["path"]))
            )

            labels_frame_map = {
                item.frame_number: item
                for item in Video.from_metaverse(video).frames
            }
            while labels_frame_map:
                ret, frame = vcap.read()

                if not ret:
                    break
                frame_id = int(vcap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_id == 38700:
                    print("This is frame_id")
                # Only write the frames for which we have done labeling.
                if str(frame_id) in labels_frame_map:
                    print("frame id in labels_frame_map")
                    frame_label = labels_frame_map[str(frame_id)]
                    for i, _ in enumerate(frame_label.actors):
                        frame_label.actors[i].confidence = 1
                    frame_label = self._pose_model(frame, frame_label)
                    for actor in frame_label.actors:

                        if actor.category == ActorCategory.PERSON:

                            # check if person is occluded, truncated or in pit
                            occluded_degree = (
                                OccludedDegree.Occluded
                                if actor.occluded
                                else (
                                    OccludedDegree[actor.occluded_degree]
                                    if actor.occluded_degree
                                    else OccludedDegree.NONE
                                )
                            )
                            operating_object = (
                                OccludedDegree.Pit
                                if actor.operating_put
                                else (
                                    OperatingObject[actor.operating_object]
                                    if actor.operating_object
                                    else OperatingObject.NONE
                                )
                            )

                            if (
                                occluded_degree != OccludedDegree.NONE
                                or operating_object != OperatingObject.NONE
                                or actor.truncated
                                or actor.occluded
                                or actor.activity is None
                            ):

                                continue

                            self._dataset_write(frame, actor, frame_id, video)

    def _dataset_write(self, frame, actor, frame_id, video):

        # write frame to google bucket
        if actor.activity.posture == PostureType.GOOD:
            klass = f"good_{self.activity_dict[self.activity_type]}"
            print("Good pose")
        elif actor.activity.posture == PostureType.BAD:
            klass = f"bad_{self.activity_dict[self.activity_type]}"
            print("Bad pose")
        else:
            klass = "random"
            print("Random")

        person_id = actor.track_id
        rect = RectangleXYWH.from_polygon(actor.polygon)

        # check if either dimension is too small
        if rect.w < 50 or rect.h < 50:
            return
        cropped_image = frame[
            rect.top_left_vertice.y : rect.top_left_vertice.y + rect.h,
            rect.top_left_vertice.x : rect.top_left_vertice.x + rect.w,
        ]

        video_uuid_flattend = video["path"].replace("/", "_")
        gcs_img_path = f"{self.output_path}/{klass}/{video_uuid_flattend}_frame_{frame_id}_{person_id}.jpg"
        print(gcs_img_path)
        print(rect.top_left_vertice.y)
        upload_cv2_image_to_gcs(
            cropped_image,
            gcs_img_path,
            storage_client=self.storage_client,
        )

        # write embedding to csv

        embedder = KeypointPoseEmbedder.from_pose(actor.pose)
        features = embedder.create_features()

        gcs_img_path = f"{self.output_path}/{klass}/{video_uuid_flattend}_frame_{frame_id}_{person_id}.jpg"

        feat_write = ""
        for feat in features:
            feat_write += str(feat[0]) + ","
            feat_write += str(feat[1]) + ","

        with open(
            "/home/nasha_voxelsafety_com/voxel/experimental/nasha/"
            + f"train_{self.activity_dict[self.activity_type]}_heuristic_test.csv",
            "a",
            encoding="UTF-8",
        ) as ft:
            ft.write(f"{gcs_img_path},{feat_write}{klass}\n")
