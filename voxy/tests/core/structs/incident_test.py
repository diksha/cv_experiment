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

from core.structs.incident import Incident


class IncidentTest(unittest.TestCase):
    def test_to_dict(self) -> None:
        """
        Tests conversion to dictionary
        """
        dummy_incident = Incident()
        self.assertTrue(dummy_incident is not None)
        dummy_incident_dict = dummy_incident.to_dict()
        converted_incident = Incident.from_dict(dummy_incident_dict)
        self.assertTrue(converted_incident is not None)
        self.assertEqual(converted_incident.to_dict(), dummy_incident_dict)


if __name__ == "__main__":
    unittest.main()
