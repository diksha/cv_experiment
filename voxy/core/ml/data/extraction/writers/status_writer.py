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

import json
import traceback as tb

from core.utils.aws_utils import upload_fileobj_to_s3


class StatusWriter:
    def __init__(self) -> None:
        self._status = {}
        self._content_type = "application/json"

    def publish_status(self, save_dir: str) -> None:
        status = json.dumps(self._status)
        file_name = "status.json"
        s3_save_path = f"s3://{save_dir}/{file_name}"
        upload_fileobj_to_s3(
            s3_save_path,
            status.encode("utf-8"),
            content_type=self._content_type,
        )

    def write_exit_status(self, exit_status: int) -> None:
        self._status["exit_status"] = exit_status

    def write_failure(
        self,
        exc_info: tuple,
    ) -> None:
        traceback = tb.format_exception(exc_info[0], exc_info[1], exc_info[2])
        traceback_string = "".join(traceback)
        self._status["traceback"] = traceback_string
