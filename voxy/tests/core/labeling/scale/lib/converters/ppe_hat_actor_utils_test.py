#
# Copyright 2020-2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import unittest
from unittest.mock import Mock, patch

import numpy as np
from scaleapi.tasks import Task

from core.labeling.scale.lib.converters.ppe_hat_actor_utils import (
    generate_consumable_labels_for_ppe_hat,
)
from core.structs.actor import ActorCategory, HeadCoveringType, OccludedDegree


class PPEHatActorUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up required ivars"""
        sample_task = Task(
            client=None,
            json={
                "task_id": "6464088837a92308e38bb9a9",
                "created_at": "2023-05-16T22:49:44.439Z",
                "completed_at": "2023-05-16T23:37:01.253Z",
                "type": "imageannotation",
                "status": "completed",
                "instruction": "",
                "params": {
                    "attachment": (
                        "s3://voxel-logs/jfe_shoji/burlington/0007/cha/"
                        "9985a846-e49a-46ee-bbad-f475335296bd/"
                        "9985a846-e49a-46ee-bbad-f475335296bd-081-mp4.png"
                    ),
                },
                "is_test": False,
                "urgency": "standard",
                "metadata": {
                    "video_uuid": (
                        "jfe_shoji/burlington/0007/cha/9985a846-e49a-46ee-bbad-f475335296bd"
                    ),
                    "relative_path": (
                        "jfe_shoji/burlington/0007/cha/9985a846-e49a-46ee-bbad-f475335296bd"
                        "/9985a846-e49a-46ee-bbad-f475335296bd-081-mp4.png"
                    ),
                    "filename": (
                        "jfe_shoji/burlington/0007/cha/9985a846-e49a-46ee-bbad-f475335296bd"
                        "_ppehat_9985a846-e49a-46ee-bbad-f475335296bd-081-mp4"
                    ),
                    "taxonomy_version": (
                        "4cc5f463432bd0798b5a493e7a2cb61c420e8580d4b1aa13b95258c23c7439f8"
                    ),
                },
                "response": {
                    "annotations": [
                        {
                            "label": "PERSON_V2",
                            "attributes": {
                                "occluded_degree": "Occluded",
                                "truncated": "False",
                                "head_covered_state": "35f77891-0f2a-426f-94c3-3f0d5a6ced99",
                            },
                            "uuid": "d5530e70-e079-48da-bdce-b6d2d36cbfb6",
                            "left": 0,
                            "top": 360,
                            "height": 101,
                            "width": 147,
                            "type": "box",
                        },
                        {
                            "label": "COVERED_HEAD",
                            "attributes": {
                                "occluded_degree": "NONE",
                                "truncated": "False",
                            },
                            "uuid": "35f77891-0f2a-426f-94c3-3f0d5a6ced99",
                            "left": 0,
                            "top": 361.29412841796875,
                            "height": 26.198676151772684,
                            "width": 42.6481491374156,
                            "type": "box",
                        },
                    ],
                    "inlineComments": [],
                    "links": [],
                    "global_attributes": {},
                    "is_customer_fix": False,
                },
            },
        )
        self.task_list = [sample_task]
        self.video_uuid = "jfe_shoji/burlington/0007/cha/9985a846-e49a-46ee-bbad-f475335296bd"

    @patch(
        "core.labeling.scale.lib.converters.ppe_hat_actor_utils.download_to_file"
    )
    @patch("core.labeling.scale.lib.converters.ppe_hat_actor_utils.cv2.imread")
    def test_generate_consumable_labels_for_ppe_hat(
        self, cv2_imread_mock: Mock, download_to_file_mock: Mock
    ) -> None:
        """Test to see output of scale to voxel conversion"""
        cv2_imread_mock.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
        download_to_file_mock.side_effect = None
        ppe_hat_labels = generate_consumable_labels_for_ppe_hat(
            self.video_uuid, self.task_list
        )
        self.assertEqual(ppe_hat_labels.uuid, self.video_uuid)
        self.assertEqual(len(ppe_hat_labels.frames), 1)
        actors = ppe_hat_labels.frames[0].actors
        self.assertEqual(len(actors), 2)
        for actor in actors:
            if actor.category == ActorCategory.PERSON_V2:
                self.assertEqual(
                    actor.head_covering_type, HeadCoveringType.COVERED_HEAD
                )
                self.assertFalse(actor.is_wearing_hard_hat)
                self.assertEqual(
                    actor.occluded_degree, OccludedDegree.Occluded
                )
                self.assertFalse(actor.truncated)
            elif actor.category == ActorCategory.COVERED_HEAD:
                self.assertEqual(actor.occluded_degree, OccludedDegree.NONE)
                self.assertFalse(actor.truncated)
            else:
                self.fail("Unexpected actor category")
