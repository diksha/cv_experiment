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

from core.execution.nodes.abstract import AbstractNode
from core.incident_machine.controller import IncidentMachineController


class IncidentMachineNode(AbstractNode):
    def __init__(self, config):

        self._controller = IncidentMachineController(
            config.get("incident").get("state_machine_monitors_requested"),
            config.get("incident").get("incident_machine_params", {}),
            config.get("camera_uuid"),
            config.get("incident").get("dry_run"),
            config.get("run_uuid"),
        )

    def process(self, state_event):
        return self._controller.process(state_event)

    def finalize(self, state_event):
        return self._controller.finalize(state_event)
