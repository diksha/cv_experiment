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

from core.structs.scenario import Scenario, ScenarioSet
from core.utils.struct_utils.scenario_utils import from_configuration_file


# trunk-ignore-all
# trunk-ignore-begin(pylint/C0115,pylint/C0116)
class ScenarioUtilsTest(unittest.TestCase):
    def test_resolves_without_error(self):
        scenario_set = from_configuration_file(
            "data/scenario_sets/regression/regression_scenarios.yaml"
        )
        self.assertIsInstance(scenario_set, ScenarioSet)
        self.assertEqual(scenario_set.name, "regression_scenarios")
        self.assertGreaterEqual(len(scenario_set.scenarios), 1)
        self.assertIsInstance(scenario_set.scenarios[0], Scenario)


# trunk-ignore-end(pylint/C0115,pylint/C0116)
