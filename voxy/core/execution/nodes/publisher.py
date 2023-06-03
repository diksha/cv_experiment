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

import os

from loguru import logger

from core.execution.nodes.abstract import AbstractNode
from core.execution.utils.graph_config_utils import (
    get_should_generate_cooldown_incidents_from_config,
)
from core.utils.publisher.publisher import Publisher


class PublisherNode(AbstractNode):
    def __init__(self, config):
        self._should_generate_cooldown_incidents = (
            get_should_generate_cooldown_incidents_from_config(config)
        )
        self._publisher = Publisher(
            portal_host=config["publisher"]["portal_host"],
            auth_token=config["publisher"]["auth_token"],
            organization_key=config["publisher"]["organization_key"],
            temp_directory=config["incident"]["temp_directory"],
            should_generate_cooldown_incidents=(
                self._should_generate_cooldown_incidents
            ),
        )
        self._enabled = config.get("publisher", {}).get("enabled", True)

    def run(self):
        if self._enabled:
            # Starts the publisher daemon
            try:
                self._publisher.run()
            # trunk-ignore(pylint/W0703)
            except Exception as e:
                logger.exception(f"Publisher Node Run: {e}")
                # trunk-ignore(pylint/W0212)
                os._exit(1)

    def finalize(self):
        if self._enabled:
            self._publisher.finalize()
