import json
import os
from abc import ABC, abstractmethod

from core.structs.actor import Actor


class DetectorHypothesis(ABC):

    _TAXONOMY_PATH = "core/labeling/scale/task_creation/taxonomies"
    BASE_ANNOTATIONS_URL = (
        "https://s3.labeling-data.net/scale-cds-public-us-west-2"
    )

    def __init__(self, video_uuid: str, project="video_playback_annotation"):
        taxonomy_path = os.path.join(self._TAXONOMY_PATH, f"{project}.json")
        with open(taxonomy_path, "r", encoding="UTF-8") as taxonomy_file:
            self._taxonomy = json.load(taxonomy_file)
            self._video_uuid = video_uuid

    def _get_frame_hypothesis(
        self, actor: Actor, frame_count: int, frame_ms: int
    ):
        """Get scale annotations from actor

        Args:
            actor (Actor): Actor output of detector and tracker
            annotations (dict): scale annotations dictionary
        """

        def attributes(actor: Actor) -> dict:
            """Get attributes from actor

            Args:
                actor (Actor): Actor to get attributes from

            Returns:
                dict: Attributes dictionary
            """
            activity_type_map = {
                "UNKWON": "None",
                "BAD": "Bad",
                "GOOD": "Good",
            }
            actor = actor.to_dict()
            activity = actor["activity"]
            if actor["operating_pit"]:
                actor["operating_object"] = "PIT"
            if activity:
                for key, value in activity.items():
                    value_store = activity_type_map[value]
                    if key == "REACHING":
                        actor["reach"] = value_store
                    if key == "LIFTING":
                        actor["bend"] = value_store
            actor.pop("activity")
            for key, value in actor.items():
                if isinstance(value, bool):
                    actor[key] = "True" if value else "False"

            actor = {k: v for k, v in actor.items() if v is not None}
            return actor

        width = (
            actor.polygon.get_bottom_right().x - actor.polygon.get_top_left().x
        )
        height = (
            actor.polygon.get_bottom_right().y - actor.polygon.get_top_left().y
        )
        frame_hypothesis = {
            "key": frame_count,
            "left": actor.polygon.get_top_left().x,
            "top": actor.polygon.get_top_left().y,
            "width": width,
            "height": height,
            "attributes": attributes(actor),
            "timestamp": frame_ms,
        }
        return frame_hypothesis

    @abstractmethod
    def process(self):
        """Process annotations from model/stored json for hypothesis"""
        raise NotImplementedError(
            "DetectorHypothesis must implement process method"
        )
