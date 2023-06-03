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
from core.perception.acausal.controller import AcausalController
from core.utils.logging.proto_node import proto_node


@proto_node(log_input=False)
class AcausalNode(AbstractNode):
    def __init__(self, config):
        self._controller = AcausalController(config=config)

    def process(self, vignette):
        return self._controller.process_vignette(vignette)

    def finalize(self):
        pass
