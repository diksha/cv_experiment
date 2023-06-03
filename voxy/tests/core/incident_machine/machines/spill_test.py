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

from core.incident_machine.machines.experimental_spill import SpillMachine
from core.structs.actor import ActorCategory
from core.structs.state import State


class SpillTest(unittest.TestCase):
    def test_state_and_events(self):
        """Test that the incident machine transitions work as expected"""
        machine = SpillMachine("", {})
        actor_id = 1
        track_uuid = "1"

        # Check if an incident is generated at 60 mins
        _state = State(
            timestamp_ms=0,
            end_timestamp_ms=3600_000,
            actor_id=actor_id,
            track_uuid=track_uuid,
            actor_category=ActorCategory.PIT,
            camera_uuid=None,
        )
        incident_list = machine.process_state_event([_state])
        self.assertEqual(len(incident_list), 1)

        # Check no incidents are generated in the next two minutes
        _state = State(
            timestamp_ms=3600_001,
            end_timestamp_ms=3602_001,
            actor_id=actor_id,
            track_uuid=track_uuid,
            actor_category=ActorCategory.PIT,
            camera_uuid=None,
        )
        incident_list = machine.process_state_event([_state])
        self.assertEqual(len(incident_list), 0)

        # Check once more incident is generated after 1 hour of first incident
        _state = State(
            timestamp_ms=7200_001,
            end_timestamp_ms=7200_003,
            actor_id=actor_id,
            track_uuid=track_uuid,
            actor_category=ActorCategory.PIT,
            camera_uuid=None,
        )
        incident_list = machine.process_state_event([_state])
        self.assertEqual(len(incident_list), 1)


if __name__ == "__main__":
    unittest.main()
