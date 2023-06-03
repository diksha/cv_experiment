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

import multiprocessing
import os

import gunicorn.app.base
from django.conf import settings

from core.portal.voxel.wsgi import application

HOST = "0.0.0.0"
PORT = os.getenv("PORT") or "8080"


def number_of_workers():
    """The number of worker processes for handling requests.

    A positive integer generally in the 2-4 x $(NUM_CORES) range:
    https://docs.gunicorn.org/en/stable/settings.html#workers

    Returns:
        int: Number of worker processes to start.
    """
    if settings.DEVELOPMENT:
        return 3
    return 1 + (multiprocessing.cpu_count() * 2)


class StandaloneApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


if __name__ == "__main__":
    options = {
        "bind": f"{HOST}:{PORT}",
        "workers": number_of_workers(),
        "timeout": 60,
    }
    StandaloneApplication(application, options).run()
