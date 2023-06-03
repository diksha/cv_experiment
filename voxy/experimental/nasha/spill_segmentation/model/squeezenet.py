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

from fastai.vision.models import squeezenet1_0, squeezenet1_1


class Squeezenet:
    """Squeezenet module"""

    def squeezenet1_0(self):
        """Squeezenet1_0 model

        Returns:
            module: squeezenet1_0
        """
        return squeezenet1_0

    def squeezenet1_1(self):
        """Squeezenet1_1 model

        Returns:
            module: squeezenet1_1
        """
        return squeezenet1_1
