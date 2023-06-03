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

from core.incident_machine.machines.no_ped_zone import NoPedZoneMachine


# trunk-ignore-all(pylint/W0212)
class BaseTest(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up test...")

    def test_active_hours(self):
        """Test that the incident machine handles active hours"""

        # Test no timestamp
        base_machine = NoPedZoneMachine("test/0001", {})
        self.assertTrue(base_machine._is_in_active_hours(1675745380000))

        # Test only start
        base_machine = NoPedZoneMachine(
            "test/0001",
            {"active_hours_start_utc": "0001-01-01T01:30:00.0Z"},
        )
        self.assertTrue(base_machine._is_in_active_hours(1675745380000))

        # Test only end
        base_machine = NoPedZoneMachine(
            "test/0001",
            {"active_hours_end_utc": "0001-01-01T05:30:00.0Z"},
        )
        self.assertTrue(base_machine._is_in_active_hours(1675745380000))

        # Test start_time < end time and in active
        base_machine = NoPedZoneMachine(
            "test/0001",
            {
                "active_hours_start_utc": "0001-01-01T01:30:00.0Z",
                "active_hours_end_utc": "0001-01-01T05:30:00.0Z",
            },
        )
        self.assertTrue(
            base_machine._is_in_active_hours(1675745380001.0)
        )  # test float conversion

        # Test start_time < end time and not in active
        base_machine = NoPedZoneMachine(
            "test/0001",
            {
                "active_hours_start_utc": "0001-01-01T01:30:00.0Z",
                "active_hours_end_utc": "0001-01-01T04:30:00.0Z",
            },
        )
        self.assertFalse(base_machine._is_in_active_hours(1675745380000))

        # Test start_time > end time and in active
        base_machine = NoPedZoneMachine(
            "test/0001",
            {
                "active_hours_start_utc": "0001-01-01T23:30:00.0Z",
                "active_hours_end_utc": "0001-01-01T05:30:00.0Z",
            },
        )
        self.assertTrue(base_machine._is_in_active_hours(1675745380000))

        # Test start_time > end time and not in active
        base_machine = NoPedZoneMachine(
            "test/0001",
            {
                "active_hours_start_utc": "0001-01-01T23:30:00.0Z",
                "active_hours_end_utc": "0001-01-01T04:30:00.0Z",
            },
        )
        self.assertFalse(base_machine._is_in_active_hours(1675745380000))

        # TODO add more tests


if __name__ == "__main__":
    unittest.main()
