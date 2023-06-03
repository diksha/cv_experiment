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

import argparse

from core.labeling.scale.lib.scale_batch_wrapper import ScaleBatchWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_name", "-b", type=str, required=True, help="Batch to cancel"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_wrapper = ScaleBatchWrapper()
    batch_wrapper.cancel_batch(args.batch_name)
