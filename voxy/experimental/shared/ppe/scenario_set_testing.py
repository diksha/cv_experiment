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
import argparse
import json
import tempfile

import cv2
import requests

from core.datasets.generators.association import PersonPpeAssociation
from core.infra.cloud.gcs_cv2_utils import upload_cv2_image_to_gcs
from core.infra.cloud.gcs_utils import (
    download_blob,
    get_storage_client,
    get_video_signed_url,
    list_blobs_with_prefix,
    separate_bucket_from_relative_path,
)
from core.labeling.label_store.label_reader import LabelReader
from core.structs.actor import ActorCategory
from core.structs.attributes import RectangleXYWH
from core.structs.video import Video
from experimental.nasha.scale_dataset import ScaleDataset


class ScenarioEval:
    def __init__(
        self,
        ppe_class=ActorCategory.SAFETY_VEST,
        no_ppe_class=ActorCategory.BARE_CHEST,
        data_path=None,
        scale_dataset_id=None,
    ):
        self.label_reader = LabelReader()
        self.ppe_class = ppe_class
        self.no_ppe_class = no_ppe_class
        self.storage_client = get_storage_client()
        self.class_dict = {
            ActorCategory.SAFETY_VEST: "vest",
            ActorCategory.BARE_CHEST: "no_vest",
            ActorCategory.HARD_HAT: "hat",
            ActorCategory.BARE_HEAD: "no_hat",
        }
        self.association = PersonPpeAssociation(
            iou_threshold=0,
            ppe_actor_class=self.ppe_class,
            no_ppe_actor_class=self.no_ppe_class,
        )
        self.data_path = data_path
        self.scale_dataset_id = scale_dataset_id

    def generate_for_video_uuid(self, video_uuid, upload_data=False):
        vcap = cv2.VideoCapture(get_video_signed_url(video_uuid))

        labels_frame_map = (
            {
                item.frame_number: item
                for item in Video.from_dict(
                    json.loads(self.label_reader.read(video_uuid))
                ).frames
            }
            if self.label_reader.read(video_uuid)
            else None
        )
        while labels_frame_map:
            ret, frame = vcap.read()

            if not ret:
                break
            frame_id = int(vcap.get(cv2.CAP_PROP_POS_FRAMES))

            # Only write the frames for which we have done labeling.
            if frame_id in labels_frame_map:
                frame_label = labels_frame_map[frame_id]

                is_ppe = self.association.get_ppe_person_association(
                    frame_label
                )
                if is_ppe is None:
                    continue

                for actor in frame_label.actors:
                    if actor.category == ActorCategory.PERSON:
                        if is_ppe.get(actor.track_id):
                            klass = f"{self.class_dict[self.ppe_class]}"
                        else:
                            klass = f"{self.class_dict[self.no_ppe_class]}"

                        person_id = actor.track_id
                        rect = RectangleXYWH.from_polygon(actor.polygon)
                        if rect.w < 50 or rect.h < 50:
                            continue
                        cropped_image = frame[
                            rect.top_left_vertice.y : rect.top_left_vertice.y
                            + rect.h,
                            rect.top_left_vertice.x : rect.top_left_vertice.x
                            + rect.w,
                        ]

                        video_uuid_flattend = video_uuid.replace("/", "_")
                        image_name = f"{video_uuid_flattend}_frame_{frame_id}_{person_id}.jpg"
                        gcs_img_path = f"{self.data_path}/{klass}/{video_uuid_flattend}_frame_{frame_id}_{person_id}.jpg"

                        # TODO : (Nasha) check whether we want to upload the dataset
                        upload_cv2_image_to_gcs(
                            cropped_image,
                            gcs_img_path,
                            storage_client=self.storage_client,
                        )
                        if upload_data:
                            self._upload_to_scale(image_name, gcs_img_path, label=f"{klass}", tax_name= f"{self.class_dict[self.ppe_class]}")

    def _upload_to_scale(
        self, image_name, gcs_img_path, label, tax_name="vest"
    ):
        TestDataset = ScaleDataset()
        # get/create the dataset
        TestDataset.get_dataset(dataset_id=self.scale_dataset_id)
        # #upload images to scale
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp:
            download_blob(gcs_img_path, temp.name)
            TestDataset.upload_to_dataset(
                image_name=image_name, image_path=temp.name
            )
        # create taxonomy if it does not exist
        TestDataset.create_taxonomy(
            labels=[
                self.class_dict[self.ppe_class],
                self.class_dict[self.no_ppe_class],
            ],
            tax_name=tax_name,
        )
        # upload ground truth labels to scale
        TestDataset.add_gt(
            image_name=image_name, label=label, tax_name=tax_name
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_uuid", "-n", type=str, required=True)
    parser.add_argument("--data_path", "-d", type=str, required=True)
    parser.add_argument("--scale_dataset_id", "-s", type=str, required=True)
    parser.add_argument("--video_uuid", "-v", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scenario_test = ScenarioEval(
        data_path=args.data_path,
        ppe_class=ActorCategory.SAFETY_VEST,
        no_ppe_class=ActorCategory.BARE_CHEST,
        scale_dataset_id=args.scale_dataset_id,
    )
    # generate data on single uuid and upload to bucket
    if args.new_uuid == "is_new":
        scenario_test.generate_for_video_uuid(
            video_uuid=args.video_uuid, upload_data=True
        )
    else:
        scenario_test.generate_for_video_uuid(
            video_uuid=args.video_uuid, upload_data=False
        )

