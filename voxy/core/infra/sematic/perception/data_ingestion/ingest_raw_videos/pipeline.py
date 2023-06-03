#
# Copyright 2022-2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

from dataclasses import dataclass
from typing import List, Optional

import sematic
from loguru import logger

from core.infra.sematic.shared.utils import PipelineSetup
from core.labeling.logs_store.chunk_and_upload_video_logs import (
    chunk_and_upload_video_logs,
    create_video_ingest_input,
)
from core.labeling.logs_store.ingest_data_collection_metaverse import (
    DataCollectionInfo,
    ingest_data_collections_to_metaverse,
)
from core.labeling.scale.runners.create_scale_tasks import (
    ScaleTaskSummary,
    create_scale_tasks,
)
from core.metaverse.api.queries import is_data_collection_in_metaverse
from core.structs.data_collection import DataCollectionType


@dataclass
class VideoIngestionSummary:
    scale_task_summary: Optional[ScaleTaskSummary]
    ingested_metaverse_videos: List[str]
    failed_ingested_metaverse_videos: List[str]


@sematic.func
def generate_video_ingest_summary(
    scale_task_summary: Optional[ScaleTaskSummary],
    ingested_metaverse_videos: List[str],
    failed_ingested_metaverse_videos: List[str],
) -> VideoIngestionSummary:
    """Generate video ingest summary object
    Args:
        scale_task_summary (Optional[ScaleTaskSummary]): output of scale ingestion
        ingested_metaverse_videos (List[str]): Successfully ingested videos to metaverse
        failed_ingested_metaverse_videos (List[str]): Failed ingested videos to metaverse
    Returns:
        VideoIngestionSummary: summary of video ingestion
    """
    return VideoIngestionSummary(
        scale_task_summary=scale_task_summary,
        ingested_metaverse_videos=ingested_metaverse_videos,
        failed_ingested_metaverse_videos=failed_ingested_metaverse_videos,
    )


@sematic.func
def pipeline(
    videos: List[str],
    is_test: bool,
    fps: float,
    metaverse_environment: str,
    pipeline_setup: PipelineSetup,
) -> VideoIngestionSummary:
    """
    The root function of the pipeline.

    Args:
        videos (List[str]): list of video uuids to ingest
        is_test (bool): labeling for test
        fps (float): desired frames per second as a decimal
        metaverse_environment (str): metaverse environment for ingestion
        pipeline_setup (PipelineSetup): setting up some defaults for pipeline
    Returns:
        VideoIngestionSummary: summary of video ingestion
    """

    # sanitize input
    videos_not_in_metaverse = []
    for video in videos:
        if not is_data_collection_in_metaverse(
            data_collection_uuid=video,
            metaverse_environment=metaverse_environment,
        ):
            videos_not_in_metaverse.append(video)
    videos = videos_not_in_metaverse

    logger.info(
        (
            "Ingesting raw videos to metaverse,"
            f"is_test,{is_test},"
            f"fps,{fps},"
            f"videos,{videos},"
        )
    )

    video_metadata = [
        DataCollectionInfo(
            data_collection_uuid=video_uuid,
            is_test=is_test,
            data_collection_type=DataCollectionType.VIDEO,
        )
        for video_uuid in videos
    ]
    chunk_and_upload_input = create_video_ingest_input(
        video_uuids=videos,
        metadata=video_metadata,
    )
    (
        chunked_video_uuids,
        to_ingest_data_collection,
    ) = chunk_and_upload_video_logs(ingest_input=chunk_and_upload_input)
    scale_summary = None
    if fps >= 0:
        scale_summary = create_scale_tasks(
            chunked_video_uuids,
            fps,
            "VideoPlaybackAnnotationTask",
            "",
            (
                "arn:aws:secretsmanager:us-west-2:203670452561:"
                "secret:scale_credentials-WHUbar"
            ),
        )
    successful_videos, failed_videos = ingest_data_collections_to_metaverse(
        to_ingest_data_collection, metaverse_environment
    )
    summary = generate_video_ingest_summary(
        scale_task_summary=scale_summary,
        ingested_metaverse_videos=successful_videos,
        failed_ingested_metaverse_videos=failed_videos,
    )
    return summary
