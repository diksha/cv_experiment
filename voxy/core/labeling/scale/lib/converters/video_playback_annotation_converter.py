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

import uuid
from typing import Callable, Dict, List, Optional

import requests
from loguru import logger
from scaleapi.tasks import Task, TaskStatus

from core.labeling.scale.lib.converters.converter_base import (
    Converter,
    VideoData,
)
from core.labeling.scale.lib.converters.polygon_utils import (
    get_polygon_vertices,
)
from core.labeling.scale.lib.scale_task_wrapper import ScaleTaskWrapper
from core.labeling.scale.registry.registry import ScaleLabelConverterRegistry
from core.labeling.video_helper_mixin import VideoHelperMixin
from core.structs.actor import Actor, HeadCoveringType
from core.structs.ergonomics import ActivityType, PostureType
from core.structs.video import Video

# trunk-ignore-all(pylint/R0903): one public method here.

PERSON_POSTURE_MAP = {
    "None": PostureType.UNKNOWN,
    "Good": PostureType.GOOD,
    "Bad": PostureType.BAD,
}


@ScaleLabelConverterRegistry.register()
class VideoPlaybackAnnotationConverter(Converter, VideoHelperMixin):
    """
    DISCLAIMER: If same video_uuid is used for another task (example: Door),
    this class overwrites the prior task data.
    """

    def __init__(
        self,
        completion_before_date: str,
        completion_after_date: str,
        project_name: str,
        credentials_arn: str,
        consumable_labels_fn: Callable = None,
    ):
        self._project_name = project_name
        super().__init__(
            completion_before_date,
            completion_after_date,
            self._project_name,
            credentials_arn,
        )

    def _get_completed_tasks(
        self,
        task_completion_after_date: str,
        task_completion_before_date: str,
    ) -> list:
        """
        Get a list of completed tasks that have been completed / updated between two dates

        Args:
            task_completion_after_date (str): earliest date task can be completed or updated to count
                for label conversion
            task_completion_before_date (str): latest date task can be completed or updated to count
                for label conversion

        Returns:
            tasks (list): list of tasks to run video label conversion
        """
        return ScaleTaskWrapper(self.credentials_arn).get_active_tasks(
            project_name=self._project_name,
            updated_after=task_completion_after_date,
            updated_before=task_completion_before_date,
            status=TaskStatus("completed"),
        )

    def _get_actor(
        self,
        track_uuid: str,
        frame_id: int,
        video_annotations: Dict[str, object],
    ) -> Dict[str, object]:
        """Creates a jsonified Actor struct from the scale frame
        Args:
            track_uuid (str): track uuid from scale identifying actor
            frame_id (int): current frame we are extracting actor info
            video_annotations (Dict[str, object]): all annotations from
                the current video
        Returns:
            Dict[str, object]: jsonified Actor struct
        """

        actor_obj = {}
        frame = video_annotations[track_uuid]["frames"][frame_id]
        actor_label = video_annotations[track_uuid]["label"]
        actor_obj["track_uuid"] = track_uuid
        actor_obj["category"] = actor_label
        if actor_label == "SIE":
            return {}
        if actor_label == "NO_HARD_HAT":
            actor_obj["category"] = "BARE_HEAD"
        if actor_label == "NO_SAFETY_VEST":
            actor_obj["category"] = "BARE_CHEST"

        polygon = {}
        polygon["vertices"] = get_polygon_vertices(frame)
        actor_obj["polygon"] = polygon
        actor_obj["uuid"] = str(uuid.uuid4())

        if frame["attributes"]:
            activity = {}
            for key_attribute in frame["attributes"].keys():
                attribute_val = frame["attributes"][key_attribute]
                if key_attribute in ("bend", "lift", "reach"):
                    if key_attribute == "bend":
                        person_bend = attribute_val
                        is_lifting = frame["attributes"]["lift"]
                        key = (
                            ActivityType.UNKNOWN.name
                            if is_lifting.lower() in ("none", "false")
                            else ActivityType.LIFTING.name
                        )
                        activity[key] = PERSON_POSTURE_MAP[person_bend].name
                    elif key_attribute == "reach":
                        person_reach = attribute_val
                        activity[
                            ActivityType.REACHING.name
                        ] = PERSON_POSTURE_MAP[person_reach].name
                elif key_attribute == "head_covered_state":
                    head_cover_type_track_uuid = frame["attributes"][
                        key_attribute
                    ]
                    head_cover_type = video_annotations[
                        head_cover_type_track_uuid
                    ]["label"]
                    actor_obj["head_covering_type"] = HeadCoveringType[
                        head_cover_type
                    ].name
                    actor_obj["is_wearing_hard_hat"] = (
                        HeadCoveringType[head_cover_type]
                        == HeadCoveringType.HARD_HAT
                    )
                elif key_attribute == "hard_hat":
                    self._get_legacy_hard_hat_info_if_required(actor_obj)
                elif key_attribute == "safety_vest":
                    actor_obj["is_wearing_safety_vest"] = True
                    actor_obj["is_wearing_safety_vest_v2"] = True
                else:
                    actor_obj[key_attribute] = (
                        attribute_val.lower() == "true"
                        if (
                            attribute_val.lower() == "false"
                            or attribute_val.lower() == "true"
                        )
                        else attribute_val
                    )

            actor_obj["activity"] = activity
        return Actor.from_dict(actor_obj)

    def _get_legacy_hard_hat_info_if_required(
        self, actor_dict: Dict[str, object]
    ):
        """Helper function to set is_wearing_hard_hat for legacy support
        Args:
            actor_dict (Dict[str, object]): actor dictionary
        """
        if actor_dict.get("is_wearing_hard_hat") is None:
            actor_dict["is_wearing_hard_hat"] = True

    def _get_video_from_task(self, task: Task) -> Optional[Video]:
        response = task.response
        video_uuid = (
            task.metadata["video_uuid"]
            if "video_uuid" in task.metadata.keys()
            else task.unique_id
        )
        # Get timestamp_ms to key_frame mapping from VideoHelperMixin.
        frame_timestamp_ms_map = self.get_frame_timestamp_ms_map(video_uuid)
        if "annotations" not in response.keys():
            return None
        response = requests.get(task.response["annotations"]["url"])
        annotations = response.json()
        frames = {}
        for key in annotations.keys():
            for frame_idx, frame in enumerate(annotations[key]["frames"]):
                actor = self._get_actor(
                    track_uuid=key,
                    frame_id=frame_idx,
                    video_annotations=annotations,
                )
                if not actor:
                    continue
                local_frame = {}
                # Annotations exist only at key frames, specified by frame["key"]. And not the
                # count of frame being annotated.
                # For .1 fps labeling for 5 fps video, key frame is 0, 50, etc.
                (timestamp_ms, height, width) = frame_timestamp_ms_map[
                    frame["key"]
                ]
                if timestamp_ms not in frames:
                    local_frame["frame_width"] = width
                    local_frame["frame_height"] = height
                    local_frame["frame_number"] = frame["key"]
                    local_frame["relative_timestamp_s"] = float(
                        timestamp_ms / 1000.0
                    )
                    local_frame["relative_timestamp_ms"] = timestamp_ms
                    local_frame["actors"] = []
                    frames[timestamp_ms] = local_frame
                else:
                    local_frame = frames[timestamp_ms]
                local_frame["actors"].append(actor.to_dict())
        final_frames = []
        for _, frame in frames.items():
            final_frames.append(frame)
        video_object = {}
        video_object["uuid"] = video_uuid
        video_object["parent_uuid"] = None
        video_object["root_uuid"] = None
        video_object["frames"] = final_frames
        return Video.from_dict(video_object)

    def _process_task(self, task: Task) -> str:
        """
        Convert task.

        Args:
            task: name of the completed batch to convert

        Returns:
            converted_video_uuid (str): video_uuid converted successfully
        """
        video = self._get_video_from_task(task)
        if not video:
            return None
        upload_successful = super().upload_consumable_labels_to_s3(
            video.uuid,
            video.to_dict(),
        )
        if upload_successful:
            return video.uuid
        logger.error("Unable to upload video")
        return None

    def convert_and_upload(self) -> List[VideoData]:
        """
        Main conversion function for all queried batches
        Returns:
            List[VideoData]: list of video uuids and metadata that have been converted
        """
        logger.info("Converting and uploading video playback annotation task")
        completed_tasks = self._get_completed_tasks(
            self.completion_after_date, self.completion_before_date
        )
        all_converted_video_uuids = []
        for task in completed_tasks:
            all_converted_video_uuids.append(
                VideoData(task.metadata, self._process_task(task))
            )
        return all_converted_video_uuids
