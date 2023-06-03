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

# trunk-ignore-all(mypy)
import json
from typing import Any

from loguru import logger

from core.utils.aws_utils import upload_fileobj_to_s3


class Logger:
    def __init__(self, name: str, config: dict) -> None:
        # defensive programming for uuid and log key
        self.video_uuid = (
            config["video_uuid"] if "video_uuid" in config else ""
        )
        self.log_key = config["log_key"] if "log_key" in config else ""
        self.id = name
        self.log_path = f"s3://voxel-temp/logs/{self.log_key}/{self.video_uuid}/{self.id}.json"
        if self.log_key:
            logger.info(f"[Logger|{self.id}] Logging enabled")
            logger.info(f"Log path: {self.log_path}")
            self.enabled = True
        else:
            logger.info(f"[Logger|{self.id}] Logging disabled")
            self.enabled = False

        self.data = []
        # Only update if the log doesn't already exist, never overwrite it.

    def log(self, serializable: Any) -> None:
        if not self.enabled:
            return
        try:
            object_dictionary = serializable.to_dict()
            # TODO: append to path in
            self.data.append(object_dictionary)

        except AttributeError:
            logger.exception("[WARNING] logger was unable to serialize object")

    def finalize(self):
        # TODO make this a GCS file handle and write it out periodically instead of
        # all at the end
        if self.enabled:
            upload_fileobj_to_s3(self.log_path, json.dumps(self.data))
