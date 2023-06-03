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

import unittest

from core.labeling.scale.hypothesis_generation.videoplayback_hypothesis import (
    annotations_hypothesis,
)


class AnnotationsHypothesisTest(unittest.TestCase):
    def test_annotations_hypothesis(self) -> None:
        video_uuid = "americold/ontario/0002/cha/20220425_01_0000"
        (
            s3_path,
            frame_rate,
            annotations,
        ) = annotations_hypothesis.AnnotationHypothesis(
            video_uuid, is_test=True
        ).process()
        print(frame_rate)
        print(annotations["38a9ab27-f6a9-4ef8-8b26-7f73585b5032"])
        self.assertEqual(frame_rate, 50)
        self.assertEqual(
            s3_path,
            "s3://voxel-datasets/hypothesis/test/americold/ontario/0002/cha/20220425_01_0000.json",
        )
        self.assertEqual(
            annotations["38a9ab27-f6a9-4ef8-8b26-7f73585b5032"],
            {
                "label": "BARE_CHEST",
                "geometry": "box",
                "frames": [
                    {
                        "key": 0,
                        "left": 673.4,
                        "top": 151.33,
                        "width": 8.389999999999986,
                        "height": 12.23999999999998,
                        "attributes": {
                            "uuid": "879b9cc2-8b9f-4895-b13e-d5f09d8f0148",
                            "category": "BARE_CHEST",
                            "polygon": {
                                "vertices": [
                                    {"x": 673.4, "y": 151.33, "z": None},
                                    {"x": 681.79, "y": 151.33, "z": None},
                                    {"x": 681.79, "y": 163.57, "z": None},
                                    {"x": 673.4, "y": 163.57, "z": None},
                                ]
                            },
                            "track_id": 225795,
                            "track_uuid": "38a9ab27-f6a9-4ef8-8b26-7f73585b5032",
                            "occluded_degree": "NONE",
                        },
                        "timestamp": 0,
                    },
                    {
                        "key": 50,
                        "left": 978.2,
                        "top": 573.72,
                        "width": 88.89999999999986,
                        "height": 97.67999999999995,
                        "attributes": {
                            "uuid": "11b0cf10-3976-4981-a5dd-1de26cdb8c50",
                            "category": "BARE_CHEST",
                            "polygon": {
                                "vertices": [
                                    {"x": 978.2, "y": 573.72, "z": None},
                                    {"x": 1067.1, "y": 573.72, "z": None},
                                    {"x": 1067.1, "y": 671.4, "z": None},
                                    {"x": 978.2, "y": 671.4, "z": None},
                                ]
                            },
                            "track_id": 225795,
                            "track_uuid": "38a9ab27-f6a9-4ef8-8b26-7f73585b5032",
                            "occluded_degree": "NONE",
                        },
                        "timestamp": 10000,
                    },
                ],
            },
        )
