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
import typing

import attr

from core.infra.symphony.api.firestore import read_from_symphony_collection


@attr.s(slots=True)
class Job:

    name: str = attr.ib()
    command: str = attr.ib()
    type: str = attr.ib()

    image: typing.Optional[str] = attr.ib(default=None)
    items: typing.Optional[typing.List[str]] = attr.ib(default=None)
    items_from_store_key: typing.Optional[str] = attr.ib(default=None)
    depends_on: typing.List[str] = attr.ib(factory=list)
    node_pool_name: typing.Optional[str] = attr.ib(default=None)

    username: str = attr.ib(default=os.environ["USER"])
    cpus: int = attr.ib(default=1)
    memory_mb: int = attr.ib(default=4000)
    gpus: int = attr.ib(default=0)
    host_ipc: bool = attr.ib(default=False)
    active_deadline_seconds: int = attr.ib(default=14400)
    allow_dependency_failure: bool = attr.ib(default=False)
    timeout_in_minutes: int = attr.ib(default=1440)
    soft_fail: bool = attr.ib(default=False)

    env: dict = attr.ib(default={})
    # Number of retries while running the k8s job.
    attempts: int = attr.ib(default=1)

    buildkite_kwargs: dict = attr.ib(default={})

    @property
    def differentiator(self):
        if self.items:
            return (self.type, self.command, tuple(sorted(self.items)))
        return (self.type, self.command, self.items_from_store_key)

    def get_items(self):
        if self.items is not None:
            return self.items
        if self.items_from_store_key is not None:
            return read_from_symphony_collection(
                os.getenv("SYMPHONY_CONFIG_FIRESTORE_UUID"),
                self.items_from_store_key,
            )
        return None

    @classmethod
    def from_dict(cls, job_config):
        if (
            job_config.get("items") is not None
            and job_config.get("items_from_store_key") is not None
        ):
            raise RuntimeError("Define only items or items_from_store_key")
        # trunk-ignore(mypy/call-arg)
        return Job(
            name=job_config["name"],
            command=job_config["command"],
            type=job_config["type"],
            image=job_config.get("image"),
            items=job_config.get("items"),
            items_from_store_key=job_config.get("items_from_store_key"),
            username=job_config.get("username") or os.environ["USER"],
            cpus=job_config.get("cpus") or 1,
            memory_mb=job_config.get("memory_mb") or 4000,
            gpus=job_config.get("gpus") or 0,
            depends_on=job_config.get("depends_on") or [],
            node_pool_name=job_config.get("node_pool_name"),
            env=job_config.get("env") or {},
            attempts=job_config.get("attempts") or 1,
            host_ipc=job_config.get("host_ipc") or False,
            active_deadline_seconds=job_config.get("active_deadline_seconds")
            or 14400,
            allow_dependency_failure=job_config.get("allow_dependency_failure")
            or False,
            timeout_in_minutes=job_config.get("timeout_in_minutes") or 1440,
            soft_fail=job_config.get("soft_fail") or False,
            buildkite_kwargs=job_config.get("buildkite_kwargs") or {},
        )

    def to_dict(self):
        return attr.asdict(self)
