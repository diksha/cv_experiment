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

from core.utils.logging.mcap import ProtoLogger


def proto_node(log_input: bool = True, log_output: bool = True):
    """
    Decorator for perception nodes
    that logs the input and output of the node conditionally

    Args:
        log_input (bool): whether to log the input of the node
        log_output (bool): whether to log the output of the node

    Returns:
        Any: the decorated node
    """

    def decorator(node: Any) -> Any:
        """
        Simple node decorator

        Args:
            node (Any): the node to decorate

        Returns:
            Any: the decorated node
        """
        node_init = node.__init__
        node_name = node.__name__
        node_process = node.process
        node_finalize = node.finalize

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            self.logger = ProtoLogger(node_name, *args, **kwargs)
            node_init(self, *args, **kwargs)

        def process(  # trunk-ignore(pylint/W9011)
            self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            if log_input:
                self.logger.log_input(*args, **kwargs)
            output = node_process(self, *args, **kwargs)
            if log_output:
                self.logger.log_output(output)
            return output

        def finalize(  # trunk-ignore(pylint/W9011)
            self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            self.logger.finalize()
            return node_finalize(self, *args, **kwargs)

        node.__init__ = __init__
        node.process = process
        node.finalize = finalize

        return node

    return decorator
