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

from core.perception.obstruction.obstruction_segmenter import (
    ObstructionSegmenter,
)
from core.perception.segmenter_tracker.tracker import SegmenterTracker


class ObstructionSegmenterTracker(unittest.TestCase):
    def test_import_segmenter(self) -> None:
        """
        Tests to see if the import failed
        """
        self.assertTrue(ObstructionSegmenter is not None)

    def test_import_tracker(self) -> None:
        """
        Tests to see if the import failed
        """
        self.assertTrue(SegmenterTracker is not None)
