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

from lib.infra.utils.resolve_model_path import resolve_all_model_paths


class AbstractGraph:
    def __init__(self, config):
        # Every graph execution must run this to ensure models are discoverable
        resolve_all_model_paths(config)
