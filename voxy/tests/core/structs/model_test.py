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

import unittest

from core.structs.model import ModelConfiguration

# proto imports fail with trunk
# trunk-ignore-begin(pylint/E0611)
from protos.perception.structs.v1 import model_pb2

# trunk-ignore-end(pylint/E0611)


class TestProtobufSchema(unittest.TestCase):
    def test_transform_messages(self):
        # test ResizeTransform message
        resize_transform = model_pb2.ResizeTransform(size=[224, 224])
        self.assertEqual(resize_transform.size, [224, 224])

        # test ToTensorTransform message
        to_tensor_transform = model_pb2.ToTensorTransform()
        self.assertTrue(
            isinstance(to_tensor_transform, model_pb2.ToTensorTransform)
        )

        # test NormalizeTransform message (image net values)
        normalize_transform = model_pb2.NormalizeTransform(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.assertAlmostEqual(normalize_transform.mean[0], 0.485, places=7)
        self.assertAlmostEqual(normalize_transform.mean[1], 0.456, places=7)
        self.assertAlmostEqual(normalize_transform.mean[2], 0.406, places=7)
        self.assertAlmostEqual(normalize_transform.std[0], 0.229, places=7)
        self.assertAlmostEqual(normalize_transform.std[1], 0.224, places=7)
        self.assertAlmostEqual(normalize_transform.std[2], 0.225, places=7)

        # test ViTFeatureExtractorTransform message
        vit_transform = model_pb2.ViTFeatureExtractorTransform(
            pretrained_model="vit-large-patch16-384"
        )
        self.assertEqual(
            vit_transform.pretrained_model, "vit-large-patch16-384"
        )

        # test Transform message
        transform = model_pb2.Transform(resize_transform=resize_transform)
        self.assertTrue(
            isinstance(transform.resize_transform, model_pb2.ResizeTransform)
        )

    def test_model_configuration_message(self):
        # test ModelConfiguration message
        model_config = model_pb2.ModelConfiguration(
            color_space=model_pb2.ColorSpace.COLOR_SPACE_RGB,
            class_index_map={"cat": 0, "dog": 1},
            preprocessing_transforms=model_pb2.TransformList(
                transforms=[
                    model_pb2.Transform(
                        resize_transform=model_pb2.ResizeTransform(
                            size=[224, 224]
                        )
                    ),
                    model_pb2.Transform(
                        to_tensor_transform=model_pb2.ToTensorTransform()
                    ),
                    model_pb2.Transform(
                        normalize_transform=model_pb2.NormalizeTransform(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        )
                    ),
                    model_pb2.Transform(
                        vit_feature_extractor_transform=model_pb2.ViTFeatureExtractorTransform(
                            pretrained_model="vit-large-patch16-384"
                        )
                    ),
                ]
            ),
            postprocessing_transforms=model_pb2.TransformList(
                transforms=[
                    model_pb2.Transform(
                        normalize_transform=model_pb2.NormalizeTransform(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        )
                    ),
                ]
            ),
        )

        self.assertEqual(
            model_config.color_space, model_pb2.ColorSpace.COLOR_SPACE_RGB
        )
        self.assertEqual(model_config.class_index_map, {"cat": 0, "dog": 1})

        self.assertEqual(
            len(model_config.preprocessing_transforms.transforms), 4
        )
        self.assertTrue(
            isinstance(
                model_config.preprocessing_transforms.transforms[
                    0
                ].resize_transform,
                model_pb2.ResizeTransform,
            )
        )
        self.assertTrue(
            isinstance(
                model_config.preprocessing_transforms.transforms[
                    1
                ].to_tensor_transform,
                model_pb2.ToTensorTransform,
            )
        )
        self.assertTrue(
            isinstance(
                model_config.preprocessing_transforms.transforms[
                    2
                ].normalize_transform,
                model_pb2.NormalizeTransform,
            )
        )
        # test creating the class
        model_config_class = ModelConfiguration(proto_message=model_config)
        self.assertTrue(isinstance(model_config_class, ModelConfiguration))
