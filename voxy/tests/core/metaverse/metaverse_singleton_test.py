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

import os
import unittest
from unittest import mock

from core.metaverse.metaverse import Metaverse


class MetaverseSingletonTest(unittest.TestCase):
    """
    Tests Metaverse singleton
    """

    # Access to a protected member __instance of a client class:
    # trunk-ignore-all(pylint/W0212)
    @staticmethod
    def __close_metaverse():
        if Metaverse._Metaverse__instance is not None:
            # We can't just call Metaverse() because some tests may have constructed it
            # in a non-default way with a specific constructor argument, which will
            # cause Metaverse.__new__ to intentionally raise exception
            Metaverse._Metaverse__instance.close()
            # Unused private member - we're just trying to garbage collect it
            # gtrunkg-ignore-all(pylint/W0238)
            Metaverse._Metaverse__instance = None

    def setUp(self):
        MetaverseSingletonTest.__close_metaverse()

    def tearDown(self):
        MetaverseSingletonTest.__close_metaverse()

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_metaverse_no_env(self) -> None:
        """Test Metaverse requires METAVERSE_ENVIRONMENT environment var"""
        with self.assertRaises(Exception) as exc_context:
            Metaverse()
        self.assertIn("METAVERSE_ENVIRONMENT", exc_context.exception.args[0])

    @mock.patch.dict(
        os.environ, {"METAVERSE_ENVIRONMENT": "INTERNAL"}, clear=True
    )
    def test_metaverse_override_env(self) -> None:
        """Test Metaverse constructor overrides env. vars"""
        mverse = Metaverse("PROD")
        self.assertEqual(mverse.environment, "PROD")

    def test_metaverse_change_env(self) -> None:
        """Test requests for conflicting Metaverse environments result in exception"""
        mverse = Metaverse("PROD")
        self.assertEqual(mverse.environment, "PROD")
        with self.assertRaises(Exception) as exc_context:
            Metaverse("INTERNAL")
        self.assertIn(
            "multiple metaverse environments", exc_context.exception.args[0]
        )


if __name__ == "__main__":
    unittest.main()
