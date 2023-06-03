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
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger
from sematic import (
    CloudResolver,
    LocalResolver,
    SilentResolver,
    api_client,
    has_container_image,
)
from sematic.abstract_future import FutureState
from sematic.future import Future


# TODO: remove this when sematic has better git hook
#  and slack notification support
#
# This is here so we can have proper notification in buildkite
#  if a sematic run has completed
# successfully
def block_until_done(run_id: str):
    """Block execution of binary until all steps are done

    Args:
        run_id (str): id of sematic run

    Raises:
        RuntimeError: If the pipeline fails
    """
    if not os.getenv("BUILDKITE_BUILD_ID"):
        logger.info("Not a buildkite pipeline, check results in sematic ui")
        return
    poll_interval_seconds = 2
    keep_going = True
    while keep_going:
        run = api_client.get_run(run_id)
        state = FutureState[run.future_state]
        keep_going = not state.is_terminal()
        time.sleep(poll_interval_seconds)
        if poll_interval_seconds <= 64:
            poll_interval_seconds = poll_interval_seconds * 2
    if state != FutureState.RESOLVED:
        raise RuntimeError(f"Run {run_id} finished in state {state}")
    logger.info("Run successful")


@dataclass
class PipelineSetup:
    buildkite_url: Optional[str]

    def __init__(self, buildkite_url=None):
        self.buildkite_url = os.getenv("BUILDKITE_BUILD_URL", default=None)

    def __setstate__(self, state):
        self.__dict__.update(state)
        logger.info(f"Buildkite url {self.buildkite_url}")


@dataclass
class SematicOptions:
    cache_namespace: Optional[str] = None
    silent: bool = False
    rerun_from: Optional[str] = None
    max_parallelism: Optional[int] = 10

    @staticmethod
    def from_args(args: Namespace) -> "SematicOptions":
        """Construct SematicOptions object from parsed commandline arguments

        Args:
            args (Namespace): Parsed commandline arguments

        Returns:
            SematicOptions: SematicOptions object created from given options
        """
        return SematicOptions(
            cache_namespace=getattr(args, "cache_namespace", None),
            silent=getattr(args, "silent", None),
            rerun_from=getattr(args, "rerun_from", None),
            max_parallelism=getattr(args, "max_parallelism", None),
        )

    @staticmethod
    def add_to_parser(parser: ArgumentParser) -> None:
        """Add sematic options to argument parser

        Args:
            parser (ArgumentParser): Argument parser to add sematic-related options
        """
        group = parser.add_argument_group(
            "Sematic", "Arguments related to Sematic orchestration"
        )
        group.add_argument(
            "--cache_namespace",
            type=str,
            default=None,
            help="Caching namespace for sematic jobs",
        )
        group.add_argument(
            "--max_parallelism",
            type=int,
            default=10,
            help="Maximum number of concurrent Sematic jobs",
        )
        group.add_argument(
            "--rerun_from",
            type=str,
            default=None,
            help=(
                "To create a new pipeline execution that picks up where a previous "
                "one left off, use this flag. The value should be a Sematic run id. "
                "The pipeline that run was in will be cloned and the execution "
                "continued from where that run was in the pipeline."
            ),
        )
        group.add_argument(
            "--silent",
            action="store_true",
            default=False,
            help="Use sematic silent resolver",
        )


def resolve_sematic_future(
    future: Future,
    options: SematicOptions,
    label: str = "",
    block_run: bool = False,
) -> Any:
    """Resolve sematic future

    Args:
        future (Future): future to resolve
        options (SematicOptions): Options for sematic resolver
        label (str, optional): Output label. Defaults to "".
        block_run: whether to block until the run is done. Defaults to False.
    Returns:
        Any: future's return value
    """
    if has_container_image():
        resolver = CloudResolver(
            cache_namespace=options.cache_namespace,
            max_parallelism=options.max_parallelism,
            rerun_from=options.rerun_from,
        )
        logger.info(f"Launching {future.id} in the cloud for {label}")
    elif options.silent:
        resolver = SilentResolver()
        logger.info(f"Launching {future.id} locally and silently for {label}")
    else:
        resolver = LocalResolver(
            cache_namespace=options.cache_namespace,
            rerun_from=options.rerun_from,
        )
        logger.info(f"Launching {future.id} locally for {label}")

    if not options.silent:
        logger.info(
            f"See results at: https://sematic.voxelplatform.com/runs/{future.id}"
        )
    resolution = future.resolve(resolver)
    if block_run:
        block_until_done(future.id)

    return resolution
