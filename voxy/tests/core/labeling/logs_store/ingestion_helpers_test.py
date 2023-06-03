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

from core.labeling.logs_store.ingestion_helpers import (
    validate_voxel_uuid_format,
)


class IngestionHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bad_voxel_uuids = [
            "americold/taunton/cha/20220101_01",
            "americold/taunton/0001/cha/20220101.01",
            "americold/taunton/0001/cha/20220101,01",
        ]

        self.good_voxel_uuids = [
            "americold/taunton/0001/cha/20220101_01",
            "americold/taunton/0001/cha/2022-01-01_01",
            "americold/modesto/0001/cha/scenarios/DURATION/negative/a1a59f61158b",
            "americold/ontario/0001/cha/test/20220422_test_01_0000",
        ]

    def test_assert_error_bad_uuids(self) -> None:
        """Tests assertions on bad uuids"""
        for voxel_uuid in self.bad_voxel_uuids:
            self.assertRaises(
                Exception, validate_voxel_uuid_format, [voxel_uuid]
            )

    def test_not_assert_error_good_uuids(self) -> None:
        """Tests no assertions on good uuids"""
        validate_voxel_uuid_format(self.good_voxel_uuids)
