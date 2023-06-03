import json
import os

import cv2
import numpy as np

from core.structs.actor import ActorCategory, draw_text
from core.utils import aws_utils
from core.utils.logging.log_decoding_utility import LogDecodingUtility
from core.utils.video_reader import S3VideoReader, S3VideoReaderInput


class StationaryVisualizer:
    @staticmethod
    def visualize(image, data, time):
        perception_frame = data["TemporalNode"]
        acausal_vignette = data["StateNode"]
        actor_id_to_tracklet = {}

        if acausal_vignette is not None:
            for tracklet_id in acausal_vignette.tracklets:
                actor_id_to_tracklet[
                    int(tracklet_id)
                ] = acausal_vignette.tracklets[tracklet_id]

        if perception_frame is not None:
            for actor in perception_frame.actors:
                StationaryVisualizer.draw_actor(
                    image,
                    actor,
                    actor_id_to_tracklet.get(actor.track_id),
                    time,
                )

    @staticmethod
    def draw_actor(image, actor, tracklet, time):
        TEXT_PADDING = 10
        if actor.category == ActorCategory.PIT:
            actor.draw(image)
            # draw some stationary info
            top_left = actor.polygon.get_top_left()
            bottom_right = actor.polygon.get_bottom_right()
            pos = (
                max(int(bottom_right.x) - TEXT_PADDING, 0),
                max(int(bottom_right.y) - TEXT_PADDING, 0),
            )
            pos_is_stationary = (
                max(int(top_left.x) + TEXT_PADDING, 0),
                max(int(top_left.y) + TEXT_PADDING, 0),
            )

            vx = actor.x_velocity_pixel_per_sec
            vy = actor.x_velocity_pixel_per_sec
            speed = np.linalg.norm([vx, vy])
            draw_text(image, f"{speed}", pos=pos, text_color=(0, 255, 0))

            if tracklet is not None:
                stationary_since = tracklet["stationary_since_ms"]
                stationary = (
                    f"{time - stationary_since} ms"
                    if stationary_since is not None
                    else "0"
                )
                moving_color = (0, 255, 0)
                stationary_color = (255, 0, 0)
                if stationary_since is not None:
                    dt = time - stationary_since
                    alpha = max(min(dt / (1000) / (30), 1), 0)
                    output_color = []
                    for moving_channel, stationary_channel in zip(
                        moving_color, stationary_color
                    ):
                        output_color.append(
                            (1 - alpha) * moving_channel
                            + alpha * stationary_channel
                        )
                    moving_color = tuple(output_color)

                draw_text(
                    image,
                    stationary,
                    pos=pos_is_stationary,
                    text_color=moving_color,
                )


class LogVisualizer:
    def __init__(self, log_key, output):
        self.log_key = log_key
        self.output = output
        self.bucket = None
        self.video_uuids = None
        self.video_uuid_to_log_map = None
        self.logs = None

        self.setup()

    def setup(self):
        """Populate log video uuid maps"""
        if aws_utils.does_s3_blob_exists(
            f"s3://voxel-temp/logs/{self.log_key}"
        ):
            self.get_logs_from_s3(f"s3://voxel-temp/logs/{self.log_key}")

    def get_logs_from_s3(self, log_path: str):
        """Populate log video uuid maps using s3 paths

        Args:
            log_path (str): path to the root directory containing logs
        """
        (
            bucket_name,
            relative_path,
        ) = aws_utils.separate_bucket_from_relative_path(log_path)
        self.logs = aws_utils.list_blobs_with_prefix(
            bucket_name, relative_path
        )
        self.bucket = bucket_name
        # make a map from uuids to video log paths
        video_uuid_to_log_map = {}
        for log in self.logs:
            log_name = log["key"]
            video_uuid = self.get_video_uuid_from_log(log_name, self.log_key)
            if video_uuid not in video_uuid_to_log_map:
                video_uuid_to_log_map[video_uuid] = []
            log_path = f"s3://voxel-temp/{log_name}"
            video_uuid_to_log_map[video_uuid].append(log_path)
        self.video_uuids = video_uuid_to_log_map.keys()
        self.video_uuid_to_log_map = video_uuid_to_log_map

    def get_video_uuid_from_log(self, log_path, log_key):
        def strip_json(path):
            return os.path.split(path)[0]

        video_uuid = strip_json(
            log_path.replace(log_key, "").replace("logs/", "")
        )
        if video_uuid[0] == "/":
            video_uuid = video_uuid[1:]
        print(video_uuid)
        return video_uuid

    def get_log(self, log_path: str) -> dict:
        """Fetches a json log stored at given path

        Args:
            log_path (str): Path to the log

        Returns:
            dict: log in dictionary format
        """
        return json.loads(aws_utils.read_from_s3(log_path))

    def get_video_reader(self, video_uuid: str) -> S3VideoReader:
        """_summary_

        Args:
            video_uuid (str): uuid referencing an unique video

        Returns:
            S3VideoReader: Video Reader Object that can be used to get frames of a video
        """
        print(f"Pulling video @ {video_uuid}")
        video_reader_input = S3VideoReaderInput(
            video_path_without_extension=video_uuid
        )
        video_reader = S3VideoReader(video_reader_input)
        return video_reader

    def draw(self, image, data, time):
        StationaryVisualizer.visualize(image, data, time)

    def visualize(self) -> None:
        """Draw the contents of the log on top of a video log"""
        for video_uuid in self.video_uuids:
            node_logs = {}

            for log in self.video_uuid_to_log_map[video_uuid]:
                log_dictionary = self.get_log(log)
                node_name = os.path.basename(log).replace(".json", "")
                node_logs[node_name] = log_dictionary

            decoded_logs = LogDecodingUtility.decode(node_logs)

            # TODO make this an actual output directory
            flattened_uuid = video_uuid.replace("/", "_")
            output_video_name = os.path.join(
                self.output, f"{flattened_uuid}.mp4"
            )
            print(output_video_name)

            writer = None
            video_reader = self.get_video_reader(video_uuid)
            for video_reader_op in video_reader.read():
                frame_ms = video_reader_op.relative_timestamp_ms
                image = video_reader_op.image
                if writer is None:
                    writer = cv2.VideoWriter(
                        output_video_name,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (image.shape[1], image.shape[0]),
                    )

                data = {
                    key: decoded_logs[key].get(frame_ms)
                    for key in decoded_logs.keys()
                }
                self.draw(image, data, frame_ms)
                writer.write(image)
            if writer is not None:
                writer.release()
            writer = None
