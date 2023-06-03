#
# Copyright 2022 Voxel Labs, Inc.
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

from core.labeling.scale.lib.utils import validate_taxonomy


class UtilsTest(unittest.TestCase):
    def test_taxonomy(self) -> None:
        """Test validating taxonomies"""
        self.assertTrue(validate_taxonomy("safety_vest_image_annotation"))
        self.assertTrue(validate_taxonomy("safety_gloves_image_annotation"))
