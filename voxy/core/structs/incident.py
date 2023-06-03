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
import uuid

import attr


@attr.s(slots=True)
class Incident:

    uuid = attr.ib(default=None, type=str)
    camera_uuid = attr.ib(default=None, type=str)
    camera_config_version = attr.ib(default=None, type=int)
    organization_key = attr.ib(default=None, type=str)
    title = attr.ib(default=None, type=str)
    incident_type_id = attr.ib(default=None, type=str)
    incident_version = attr.ib(default=None, type=str)
    priority = attr.ib(default=None, type=str)
    actor_ids = attr.ib(factory=list)

    # TODO(PRO-558): Deprecate GCS
    video_thumbnail_gcs_path = attr.ib(default=None, type=str)
    video_gcs_path = attr.ib(default=None, type=str)
    original_video_gcs_path = attr.ib(default=None, type=str)
    annotations_gcs_path = attr.ib(default=None, type=str)

    video_thumbnail_s3_path = attr.ib(default=None, type=str)
    video_s3_path = attr.ib(default=None, type=str)
    original_video_s3_path = attr.ib(default=None, type=str)
    annotations_s3_path = attr.ib(default=None, type=str)
    start_frame_relative_ms = attr.ib(default=None, type=float)
    end_frame_relative_ms = attr.ib(default=None, type=float)
    incident_group_start_time_ms = attr.ib(default=None, type=float)

    # Time for which add frames before and after the incident to allow context.
    pre_start_buffer_ms = attr.ib(default=None, type=float)
    post_end_buffer_ms = attr.ib(default=None, type=float)
    docker_image_tag = attr.ib(default=os.getenv("IMAGE_TAG"), type=str)

    # Cooldown tag
    cooldown_tag = attr.ib(default=False, type=bool)
    track_uuid = attr.ib(default=None, type=str)
    sequence_id = attr.ib(default=None, type=int)
    run_uuid = attr.ib(default=None, type=str)

    # List of tail incident UUIDS that were aggregated to create this incident
    tail_incident_uuids = attr.ib(factory=list, type=list)

    def __attrs_post_init__(self):
        self.uuid = self.uuid or str(uuid.uuid4())

    def to_dict(self):
        return {
            "uuid": self.uuid,
            "camera_uuid": self.camera_uuid,
            "camera_config_version": self.camera_config_version,
            "organization_key": self.organization_key,
            "title": self.title,
            "incident_type_id": self.incident_type_id,
            "incident_version": self.incident_version,
            "priority": self.priority,
            "actor_ids": self.actor_ids,
            "video_thumbnail_gcs_path": self.video_thumbnail_gcs_path,
            "video_thumbnail_s3_path": self.video_thumbnail_s3_path,
            "video_gcs_path": self.video_gcs_path,
            "video_s3_path": self.video_s3_path,
            "original_video_gcs_path": self.original_video_gcs_path,
            "original_video_s3_path": self.original_video_s3_path,
            "annotations_gcs_path": self.annotations_gcs_path,
            "annotations_s3_path": self.annotations_s3_path,
            "start_frame_relative_ms": self.start_frame_relative_ms,
            "end_frame_relative_ms": self.end_frame_relative_ms,
            "incident_group_start_time_ms": self.incident_group_start_time_ms,
            "pre_start_buffer_ms": self.pre_start_buffer_ms,
            "post_end_buffer_ms": self.post_end_buffer_ms,
            "docker_image_tag": self.docker_image_tag,
            "cooldown_tag": self.cooldown_tag,
            "track_uuid": self.track_uuid,
            "sequence_id": self.sequence_id,
            "run_uuid": self.run_uuid,
            "tail_incident_uuids": self.tail_incident_uuids,
        }

    @classmethod
    def from_dict(self, data):
        return Incident(
            uuid=data.get("uuid", None),
            camera_uuid=data.get("camera_uuid", None),
            camera_config_version=data.get("camera_config_version", None),
            organization_key=data.get("organization_key", None),
            title=data.get("title", None),
            incident_type_id=data.get("incident_type_id", None),
            incident_version=data.get("incident_version", None),
            priority=data.get("priority", None),
            actor_ids=data.get("actor_ids", None),
            video_thumbnail_gcs_path=data.get(
                "video_thumbnail_gcs_path", None
            ),
            video_thumbnail_s3_path=data.get("video_thumbnail_s3_path", None),
            video_gcs_path=data.get("video_gcs_path", None),
            video_s3_path=data.get("video_s3_path", None),
            original_video_gcs_path=data.get("original_video_gcs_path", None),
            original_video_s3_path=data.get("original_video_s3_path", None),
            annotations_gcs_path=data.get("annotations_gcs_path", None),
            annotations_s3_path=data.get("annotations_s3_path", None),
            start_frame_relative_ms=data.get("start_frame_relative_ms", None),
            end_frame_relative_ms=data.get("end_frame_relative_ms", None),
            incident_group_start_time_ms=data.get(
                "incident_group_start_time_ms", None
            ),
            pre_start_buffer_ms=data.get("pre_start_buffer_ms", None),
            post_end_buffer_ms=data.get("post_end_buffer_ms", None),
            docker_image_tag=data.get("docker_image_tag", None),
            cooldown_tag=data.get("cooldown_tag", False),
            track_uuid=data.get("track_uuid"),
            sequence_id=data.get("sequence_id"),
            run_uuid=data.get("run_uuid"),
            tail_incident_uuids=data.get("tail_incident_uuids", None),
        )
