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

from typing import Any

from core.utils.logging.logger import Logger


def node_logger(node: Any) -> Any:
    node_init = node.__init__
    node_name = node.__name__
    node_process = node.process
    node_finalize = node.finalize

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        self.logger = Logger(node_name, *args, **kwargs)
        node_init(self, *args, **kwargs)

    def process(self: Any, *args: Any, **kwargs: Any) -> Any:
        self.logger.log(*args, **kwargs)
        return node_process(self, *args, **kwargs)

    def finalize(self: Any, *args: Any, **kwargs: Any) -> Any:
        self.logger.finalize()
        return node_finalize(self, *args, **kwargs)

    node.__init__ = __init__
    node.process = process
    node.finalize = finalize

    return node
