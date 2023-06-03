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
from unittest.mock import MagicMock, Mock, patch

from core.metaverse.api.data_collection_queries import (
    get_or_create_camera_uuid,
)


class MetaverseSingletonTest(unittest.TestCase):
    @patch("core.metaverse.api.data_collection_queries.Metaverse")
    def test_get_or_create_camera_uuid(self, mock_metaverse: Mock) -> None:
        """Tests get_or_create_camera_uuid function

        Args:
            mock_metaverse (Mock): mocking metaverse execution
        """
        mock_metaverse_responses = [MagicMock(), MagicMock()]
        camera_result, camera_create_result = (MagicMock(), MagicMock())
        camera_result.data = {"camera": None}
        camera_create_result.data = {
            "camera_create": {"camera": {"uuid": "some_uuid"}}
        }
        mock_metaverse_responses[0].schema.execute.return_value = camera_result
        mock_metaverse_responses[
            1
        ].schema.execute.return_value = camera_create_result
        mock_metaverse.side_effect = mock_metaverse_responses
        self.assertEqual(
            get_or_create_camera_uuid("americold/modesto/0001/cha"),
            "some_uuid",
        )
