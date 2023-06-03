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
"""Script to generate a signed URL for a GCS object."""
import argparse

from core.labeling.video_helper_mixin import VideoHelperMixin


def parse_args():
    parser = argparse.ArgumentParser()
    # object_path is relative to --bucket
    parser.add_argument("--object_path", type=str, required=True)
    parser.add_argument("--bucket", default="voxel-videos", type=str)
    parser.add_argument("--expiration_minutes", default=30, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    args = parse_args()
    helper = VideoHelperMixin()
    url = helper.get_video_url(
        args.object_path,
        bucket=args.bucket,
        expiration_minutes=args.expiration_minutes,
    )
    print(url)
