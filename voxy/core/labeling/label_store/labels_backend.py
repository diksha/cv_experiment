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
from core.labeling.label_store.label_reader import LabelReader
from core.labeling.label_store.label_writer import LabelWriter


class LabelsBackend:
    def __init__(
        self,
        consumable=True,
        label_format="json",
        version="v1",
        project="sodium-carving-227300",
    ):
        bucket = "voxel-consumable-labels"
        provider = "s3"
        if not consumable:
            bucket = "voxel-raw-labels"
            version = ""
            provider = "gcs"
        self._reader = LabelReader(
            label_format=label_format,
            version=version,
            bucket=bucket,
            project=project,
            provider=provider,
        )
        self._writer = LabelWriter(
            label_format=label_format,
            version=version,
            bucket=bucket,
            project=project,
        )

    def label_reader(self):
        return self._reader

    def label_writer(self):
        return self._writer
