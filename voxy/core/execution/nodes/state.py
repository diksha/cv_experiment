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
from core.state.controller import StateController


class StateNode(AbstractNode):
    def __init__(self, config, otlp_meter):
        self._controller = StateController(
            config=config, otlp_meter=otlp_meter
        )

    # trunk-ignore(pylint/W9011)
    # trunk-ignore(pylint/W9012)
    def process(self, vignette):
        return self._controller.process_vignette(vignette)

    # trunk-ignore(pylint/W9011)
    # trunk-ignore(pylint/W9012)
    def finalize(self):
        return self._controller.finalize()
