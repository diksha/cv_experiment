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

import re


def validate_voxel_uuid_format(videos: list) -> None:
    """
    Args:
        videos: list of human readable voxel video uuids

    Raises:
        RuntimeError: if voxel_uuid format is not followed
        RuntimeError: if voxel_uuid contains special characters
    """
    for video_uuid in videos:
        chars = "[a-zA-Z0-9-_]+"
        reg_exp_scenarios = f"{chars}/{chars}/{chars}/{chars}/scenarios/{chars}/{chars}/{chars}$"
        reg_exp_test = f"{chars}/{chars}/{chars}/{chars}/test/{chars}$"
        reg_exp_door = f"{chars}/{chars}/{chars}/{chars}/door/{chars}$"
        reg_exp_detector = f"{chars}/{chars}/{chars}/{chars}/detector/{chars}$"
        reg_exp_door_test = (
            f"{chars}/{chars}/{chars}/{chars}/door/test/{chars}$"
        )
        reg_exp_train = f"{chars}/{chars}/{chars}/{chars}/{chars}$"
        if not (
            bool(re.match(reg_exp_scenarios, video_uuid))
            or bool(re.match(reg_exp_test, video_uuid))
            or bool(re.match(reg_exp_door, video_uuid))
            or bool(re.match(reg_exp_door_test, video_uuid))
            or bool(re.match(reg_exp_train, video_uuid))
            or bool(re.match(reg_exp_detector, video_uuid))
        ):
            raise RuntimeError(
                f"Video UUID does not follow the naming convention {video_uuid}"
            )
