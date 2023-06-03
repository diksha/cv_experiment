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

from core.structs.actor import ActorCategory


def generate_image_filename(timestamp_ms: int) -> str:
    return f"frame_{timestamp_ms}.jpg"


def get_all_actor_classes() -> list:
    return list(ActorCategory)
