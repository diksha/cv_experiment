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
from torchvision.models import resnet


class Resnet:
    """
    Class defining resnet models
    """

    def resnet18(self):
        """Resnet 18 model

        Returns:
            module: resnet18 module
        """
        return resnet.resnet18

    def resnet34(self):
        """Resnet 34 model

        Returns:
            module: resnet34 module
        """
        return resnet.resnet34

    def resnet50(self):
        """Resnet 50 model

        Returns:
            module: resnet50 module
        """
        return resnet.resnet50

    def resnet101(self):
        """Resnet 101 model

        Returns:
            module: resnet101 module
        """
        return resnet.resnet101

    def resnet152(self):
        """Resnet 152 model

        Returns:
            module: resnet152 module
        """
        return resnet.resnet152
