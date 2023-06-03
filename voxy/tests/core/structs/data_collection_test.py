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

from core.structs.data_collection import DataCollection
from core.structs.frame import Frame


class DataCollectionTest(unittest.TestCase):
    def test_data_collection(self) -> None:
        """Test creating data collection from dictionary"""
        data_collection_dict = {
            "gcs_path": "gcs_path",
            "frame_ref": [{"actors_ref": []}],
        }
        data_collection = DataCollection(**data_collection_dict)
        self.assertEqual(data_collection.to_dict(), data_collection_dict)
        self.assertEqual(
            data_collection.frames,
            [
                Frame(
                    frame_number=None,
                    frame_width=None,
                    frame_height=None,
                    relative_timestamp_s=None,
                    relative_timestamp_ms=None,
                    epoch_timestamp_ms=None,
                    actors=[],
                )
            ],
        )
