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
from experimental.nasha.dataset_gen.person_pose_classification import (
    DatasetGeneratorPersonPoseClassification,
)
import argparse
from core.structs.ergonomics import ActivityType

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--customer", "-c", type=str, required=True
    )

    parser.add_argument(
        "--dataset_version", "-d", type=str, required=True
    )
    parser.add_argument(
        "--video_reference", "-v", type=str, required=True
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dataset_output_path = (
        f"gs://voxel-users/nasha/pose_classification/dataset_{args.customer}/v{args.dataset_version}"
    )

    # Complete Dataset    
    dataset_generator = DatasetGeneratorPersonPoseClassification(f"{base_dataset_output_path}",data_source=args.customer,activity_type=ActivityType.LIFTING)
    dataset_generator.generate_for_video_uuid(args.video_reference)
