# #
# # Copyright 2020-2021 Voxel Labs, Inc.
# # All rights reserved.
# #
# # This document may not be reproduced, republished, distributed, transmitted,
# # displayed, broadcast or otherwise exploited in any manner without the express
# # prior written permission of Voxel Labs, Inc. The receipt or possession of this
# # document does not convey any rights to reproduce, disclose, or distribute its
# # contents, or to manufacture, use, or sell anything that it may describe, in
# # whole or in part.
# #
import unittest
from datetime import datetime
from unittest.mock import patch

from sematic.resolvers.silent_resolver import SilentResolver

from core.infra.sematic.perception.labeling.convert_scale_to_voxel.pipeline import (
    pipeline,
)
from core.infra.sematic.shared.utils import PipelineSetup


# trunk-ignore-begin(pylint/C0116,pylint/E1101)
class PipelineTest(unittest.TestCase):
    @patch(
        "core.infra.sematic.perception.labeling."
        "convert_scale_to_voxel.pipeline."
        "convert_scale_to_voxel_sematic_wrapper"
    )
    def test_pipeline(self, mock_convert_scale_to_voxel_sematic_wrapper):
        mock_convert_scale_to_voxel_sematic_wrapper.return_value = []
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        result = pipeline(
            now,
            0,
            "ImageCollectionAnnotationConverter",
            "generate_consumable_labels_for_doors",
            "door_state_classification",
            "scale_credentials",
            pipeline_setup=PipelineSetup(),
        ).resolve(SilentResolver())
        self.assertEqual(result, True)


# trunk-ignore-end(pylint/C0116,pylint/E1101)
