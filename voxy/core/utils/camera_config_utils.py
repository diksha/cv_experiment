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

import typing


def get_camera_uuids_from_organization_site(
    organization: str, location: typing.Optional[str] = None
) -> typing.List[str]:
    """
    Returns all cameras that have the given organization and site defined.
    If location is None, then it will return all cameras for a particular organization

    This internally uses this file:

    `configs/cameras/cameras`

    https://github.com/voxel-ai/voxel/blob/main/configs/cameras/cameras

    Args:
        organization (str): the organization to grab cameras for
        location (typing.Optional[str]): the optional location to grab cameras for

    Returns:
        typing.List[str]: the list of camera uuids
    """

    # open the text file
    with open("configs/cameras/cameras") as file:
        raw_lines = file.readlines()
        configs = [line.rstrip() for line in raw_lines]

    # filter
    camera_uuids = [
        "/".join(config.split("/")[2:]).replace(".yaml", "")
        for config in configs
    ]

    organization_uuids = [
        camera_uuid
        for camera_uuid in camera_uuids
        if camera_uuid.split("/")[0] == organization
    ]
    if location is None:
        return organization_uuids

    return [
        camera_uuid
        for camera_uuid in organization_uuids
        if camera_uuid.split("/")[1] == location
    ]
