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
import cv2

from core.infra.cloud.gcs_utils import dump_to_gcs


def upload_cv2_image_to_gcs(
    frame, gcs_path, project="sodium-carving-227300", storage_client=None
):
    return dump_to_gcs(
        gcs_path,
        cv2.imencode(".jpg", frame)[1].tostring(),
        content_type="image/jpeg",
        storage_client=storage_client,
    )
