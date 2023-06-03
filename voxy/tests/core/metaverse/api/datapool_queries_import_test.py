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

import os
import unittest
from unittest.mock import Mock, patch

# This needs to be before the next import to avoid a network call on import
os.environ["LIGHTLY_DID_VERSION_CHECK"] = "True"

# trunk-ignore(pylint/C0413,flake8/E402)
from core.metaverse.api.datapool_queries import DatapoolQueryException


class DatapoolQueriesTest(unittest.TestCase):
    @patch("core.metaverse.api.datapool_queries.Metaverse")
    def test_import(self, mock_metaverse: Mock) -> None:
        """Simple test that makes sure we can import and load

        Args:
          mock_metaverse (Mock): metaverse mock
        """
        # create object
        DatapoolQueryException()
