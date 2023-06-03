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

# Unused imports required for registry to register all functions.
#
# trunk-ignore-all(pylint/W0611)
# trunk-ignore-all(flake8/F401)

"""Registers all the functions in their respective registry that can
be used by the config driven classes to call the respective functions.

Example: Transforms are registered in TransformerRegistry. Config like the following:
transforms:
      - name: CropFromActor
        params: {}
will look for CropFromActor class registered in TransformerRegistry and call that.
"""

import core.common.functional.transforms.transforms
import core.ml.data.generation.resources.writers.csv_writer
import core.ml.data.generation.resources.writers.yolo_writer
from core.ml.data.generation.resources.logset_generators import (
    logset_generators,
)
from core.ml.data.generation.resources.streams import (
    metaverse_stream,
    synchronized_readers,
)
