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

import unittest

from core.perception.inference.transforms.registry import REGISTRY


class RegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_registry_map(self) -> None:
        """Tests the registry map and no string errors."""
        for key, value in REGISTRY.items():
            assert key == value.__name__
