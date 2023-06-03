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
from core.execution.nodes.acausal import AcausalNode
from core.execution.nodes.incident_machine import IncidentMachineNode
from core.execution.nodes.state import StateNode
from core.execution.nodes.temporal import TemporalNode

# trunk-ignore(pylint/W0105)
"""
This node is run only in Production graph currently. This node
runs the whole of Perception and state/incident system in one because
this gives us significant improvement in run times by reducing the total
time for serialization.
"""


class MegaNode(AbstractNode):
    def __init__(self, config, otlp_meter=None):
        self._temporal_node = TemporalNode(config)
        self._acausal_node = AcausalNode(config)
        self._state_node = StateNode(config, otlp_meter)
        self._incident_machine_node = IncidentMachineNode(config)

    def process(self, current_frame_struct):
        """Process the current frame struct to produce a list of incidents

        Args:
            current_frame_struct (Any): a frame struct

        Returns:
            list: incidents
        """
        incidents = []
        vignette = self._temporal_node.process(current_frame_struct)
        vignette = self._acausal_node.process(vignette)
        state_messages = self._state_node.process(vignette)
        incidents.extend(self._incident_machine_node.process(state_messages))
        return incidents

    # trunk-ignore(pylint/W9011)
    # trunk-ignore(pylint/W9012)
    def finalize(self):
        self._temporal_node.finalize()
        self._acausal_node.finalize()
        state_messages = self._state_node.finalize()
        return self._incident_machine_node.finalize(state_messages)
