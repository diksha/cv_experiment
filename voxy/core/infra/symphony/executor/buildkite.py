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

import os

from pybuildkite.buildkite import Buildkite


class BuildkiteExecutor:
    @staticmethod
    def get_buildkite_access_token():
        buildkite_file_name = f"{os.getenv('HOME')}/.buildkite/accessToken"
        if not os.path.exists(buildkite_file_name):
            raise RuntimeError(
                "Please add buildkite accesstoken to file ~/.buildkite/accessToken"
            )
        return open(buildkite_file_name).read().strip()

    @staticmethod
    def execute(config_firestore_uuid, config_path, branch, commitsha):
        buildkite = Buildkite()
        buildkite.set_access_token(
            BuildkiteExecutor.get_buildkite_access_token()
        )
        response = buildkite.builds().create_build(
            organization="voxel",
            pipeline="symphony",
            commit=commitsha,
            branch=branch,
            env={
                "SYMPHONY_CONFIG_FIRESTORE_UUID": config_firestore_uuid,
                "USER": os.environ["USER"],
            },
            clean_checkout=True,
            message=f"Running {config_path} on {branch}:{commitsha}",
            ignore_pipeline_branch_filters=True,
        )
        print(f"Build created at {response['web_url']}")
