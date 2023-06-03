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

"""Module for constants that may change but which are used in multiple build files"""

# When running Sematic pipelines that execute in the cloud,
# the code for the pipeline execution will be packaged as a
# docker image and pushed to this container registry, where
# it will be pulled from by the cloud workers.
container_registry_for_sematic_push = "203670452561.dkr.ecr.us-west-2.amazonaws.com"
