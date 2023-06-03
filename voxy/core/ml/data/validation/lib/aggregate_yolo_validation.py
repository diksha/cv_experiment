#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import os
import re

import pandas as pd
from tqdm import tqdm

from core.utils.aws_utils import (
    glob_from_bucket,
    read_decoded_bytes_from_s3,
    upload_fileobj_to_s3,
)


def generate_validation_csv(bucket: str, relative_path: str) -> str:
    """
    Generates an aggregate validation csv with all the splits in the validation directory
    after an automated YOLO run
    Args:
        bucket(str): bucket where validation results are located
        relative_path(str): relative path to validation results

    Returns:
        path to uploaded CSV
    """
    val_output = [
        os.path.join(f"s3://{bucket}", rel_path)
        for rel_path in glob_from_bucket(bucket, relative_path, ("txt"))
        if "logging_out.txt" in rel_path
    ]
    val_aggregate = pd.DataFrame(
        columns=[
            "split",
            "class",
            "images",
            "labels",
            "precision",
            "recall",
            "map_0p5",
            "map_0p5_0p95",
        ]
    )
    for val_output_path in tqdm(val_output):
        perf = read_decoded_bytes_from_s3(val_output_path).split("\n")
        split = val_output_path.split("/")[-2]
        for line in perf[1:]:
            data = [split] + re.sub(r"\s+", ",", line.strip()).split(",")
            val_aggregate.loc[len(val_aggregate)] = data
    s3_path = os.path.join(
        f"s3://{bucket}/{relative_path}", "val_aggregate.csv"
    )
    upload_fileobj_to_s3(
        s3_path,
        val_aggregate.to_csv(index=False).encode("utf-8"),
        "text/csv",
    )
    return s3_path
