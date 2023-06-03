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

from core.infra.symphony.utils.config import (
    get_job_args,
    load_job_struct_from_firestore,
)


class AbstractJob:

    bazel_target_name = None
    job_type = None

    def __init__(self, job_struct):
        self._job_struct = job_struct
        for field_name in self.additional_required_fields:
            if getattr(self._job_struct, field_name, None) is None:
                raise ValueError(f"{field_name} is a required field")

    @property
    def additional_required_fields(self):
        return []

    def execute(self):
        """
        Exeucte the command if ran as binary.
        """
        raise NotImplementedError()

    def get_run_command(self, firestore_path, job_idx):
        return [
            "./bazel",
            "run",
            f"//core/infra/symphony/job:{self.bazel_target_name}",
            f"{firestore_path}/{job_idx}",
        ]

    @staticmethod
    def parse_args_and_get_job_struct():
        return load_job_struct_from_firestore(get_job_args().firestore_path)

    @property
    def job_struct(self):
        return self._job_struct
