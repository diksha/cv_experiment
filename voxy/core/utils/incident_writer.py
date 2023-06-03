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
"""Daemon which publishes incidents downstream as they become available.

Run with:
    ./bin/python core/incidients/publisher.py
"""
import copy
import json
import os
import queue
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Tuple, Union

import av
from loguru import logger

from core.infra.cloud import kinesis_utils
from core.structs.incident import Incident
from core.structs.video import Video
from core.utils.bounded_sorted_dict import BoundedSortedDict


@dataclass
class IncidentWriterIncidents:
    incidents: List[Incident]


@dataclass
class IncidentWriterFrame:
    timestamp: int
    frame_and_struct: tuple


@dataclass(order=True)
class IncidentWriterIncident:
    sort_index: float = field(init=False, repr=False)
    incident: Incident
    end_time: float

    def __post_init__(self):
        self.sort_index = self.end_time


@dataclass
class IncidentWriterInput:
    temp_directory: str = ""
    dry_run: bool = False
    camera_arn: Optional[str] = None
    video_uuid: Optional[str] = None
    fps: Optional[float] = None
    should_generate_cooldown_incidents: bool = False
    kvs_read_session: Optional[any] = None
    frame_struct_buffer_size: int = 2000


class IncidentWriter:

    _INCIDENT_VIDEO_FPS = (
        10  # TODO(Vai): remove hardcode and read as parameter
    )

    def __init__(self, input_params: IncidentWriterInput) -> None:
        """
        Args:
            kvs_read_session (Optional[boto3.Session], optional):
                boto3 session for reading from kinesis video streams.
                Defaults to None.
        """
        self._temp_directory = input_params.temp_directory
        self._dry_run = input_params.dry_run
        self._last_frame_time_ms = 0
        self._incident_list: List[IncidentWriterIncident] = []
        self._frame_and_struct_buffer = BoundedSortedDict(
            input_params.frame_struct_buffer_size
        )
        self._frame_size: Optional[Tuple[float, float]] = None
        self._data_queue: queue.Queue[
            Union[IncidentWriterIncidents, IncidentWriterFrame]
        ] = queue.Queue()
        self.video_uuid = input_params.video_uuid
        self.camera_arn = input_params.camera_arn
        self._fps = input_params.fps
        self._should_generate_cooldown_incidents = (
            input_params.should_generate_cooldown_incidents
        )
        self._kvs_read_session = input_params.kvs_read_session

    def process_next(self, finalize: bool = False) -> None:
        """Processes at least one queued frame or incident, blocking until one is available in the queue

        This function is not threadsafe and must be called from a single thread.

        Args:
            finalize (bool): causes this function to retrieve all data from the queue and process all incidents without blocking for new data
        """

        while finalize:
            try:
                self._process(self._data_queue.get(block=False))
            except queue.Empty:
                self._process(None, finalize=True)
                return

        self._process(self._data_queue.get())

    def _process(
        self,
        data: Optional[Union[IncidentWriterIncidents, IncidentWriterFrame]],
        finalize: bool = False,
    ) -> None:
        # list of incidents to process
        incidents_to_process = []

        if isinstance(data, IncidentWriterIncidents):
            for inc in data.incidents:
                # first calculate the end time for all new incidents
                end_time = inc.end_frame_relative_ms
                if inc.post_end_buffer_ms:
                    end_time += inc.post_end_buffer_ms

                # if the incident end_time is beyond the current frame push it onto the queue, otherwise process the incident
                if end_time > self._last_frame_time_ms and not finalize:
                    self._incident_list.append(
                        IncidentWriterIncident(incident=inc, end_time=end_time)
                    )
                else:
                    incidents_to_process.append(inc)

            # sort the incident queue after any insertions
            self._incident_list.sort()

        # if we got a frame we pop it on to the frame buffer
        if isinstance(data, IncidentWriterFrame):
            self._last_frame_time_ms = data.timestamp
            self._frame_and_struct_buffer[
                data.timestamp
            ] = data.frame_and_struct

        while len(self._incident_list) > 0 and (
            self._incident_list[0].end_time < self._last_frame_time_ms
            or finalize
        ):
            inc = self._incident_list[0].incident
            self._incident_list = self._incident_list[1:]
            incidents_to_process.append(inc)

        if len(incidents_to_process) > 0:
            self._write_incidents_to_disk(incidents_to_process)

    def finalize(self) -> None:
        self.process_next(finalize=True)

    def _set_frame_size(self, frame_size: Tuple[float, float]) -> None:
        self._frame_size = frame_size

    def insert_frame_and_struct(
        self, timestamp: int, frame_and_struct: tuple
    ) -> None:
        """Inserts frame and struct data for incident writer processing, can be called from any thread"""
        self._data_queue.put(
            IncidentWriterFrame(
                timestamp=timestamp, frame_and_struct=frame_and_struct
            )
        )

    def add_incident(self, incidents: List[Incident]) -> None:
        """Inserts incident data for incident writer processing, can be called from any thread"""
        if not self._dry_run:
            self._data_queue.put(IncidentWriterIncidents(incidents=incidents))
        else:
            if len(incidents) > 0:
                # TODO: pretty print incident details
                print(
                    "Found",
                    str(len(incidents)),
                    "incidents. Did not write to disk as dry_run is enabled.",
                )

    def _write_incidents_to_disk(self, incidents: List[Incident]) -> None:
        """Writes incident data and assets to the local disk.

        Produces the following files:
            /<uuid>_incident.json
            /<uuid>_annotations.json
            /<uuid>_thumbnail.jpg
            /<uuid>_video.webm
            /<uuid>_log.mcap

        Args:
            incidents (list): List of incidents to write to disk.
        """
        for incident in incidents:
            if (
                not incident.cooldown_tag
                or self._should_generate_cooldown_incidents
            ):
                # Important: Write incident JSON last since this triggers the publisher
                # Else we could have incomplete video files being uploaded.
                video_struct, sorted_timestamps = self._generate_video_struct(
                    incident
                )
                video_filepath = self._write_incident_video(
                    incident, video_struct, sorted_timestamps
                )
                self._write_incident_thumbnail(
                    incident, video_filepath=video_filepath
                )
                self._write_incident_annotations(incident, video_struct)
                self._write_mcap_log(incident, video_struct)

            # Write incident regardless of cooldown
            self._write_incident(incident)

    def _write_incident_thumbnail(
        self, incident: Incident, video_filepath: str = None
    ) -> None:
        filename = f"{incident.uuid}_thumbnail.jpg"
        filepath = os.path.join(self._temp_directory, filename)
        kinesis_utils.extract_thumbnail(
            inputpath=video_filepath, outputpath=filepath
        )

    def _write_incident_video(
        self, incident: Incident, video_struct: Video, sorted_timestamps: List
    ) -> str:
        temp_filename = f"{incident.uuid}_video.mp4"
        temp_original_filename = f"{incident.uuid}_original_video.mp4"

        temp_filepath = os.path.join(self._temp_directory, temp_filename)
        temp_original_filepath = os.path.join(
            self._temp_directory, temp_original_filename
        )
        if self.video_uuid is not None:
            self._write_incident_from_frame_buffer(
                temp_filepath, incident, video_struct
            )
        elif self.camera_arn is not None:
            if self._fps is None:
                raise RuntimeError(
                    "kinesis_utils.download_media requires fps to be set"
                )
            kinesis_utils.download_media(
                camera_arn=self.camera_arn,
                filepath=temp_filepath,
                sorted_timestamps=sorted_timestamps,
                fps=self._fps,
                session=self._kvs_read_session,
            )
            kinesis_utils.download_original_video(
                camera_arn=self.camera_arn,
                filepath=temp_original_filepath,
                sorted_timestamps=sorted_timestamps,
                session=self._kvs_read_session,
            )
        else:
            raise RuntimeError(
                "Incident video can't be generated. Specify video_uuid or camera_arn"
            )
        return temp_filepath

    def _write_incident_from_frame_buffer(
        self, temp_filepath: str, incident: Incident, video_struct: Video
    ) -> None:
        if not self._frame_size:
            logger.info("Frame size is not set. Setting now.")
            (height, width, _,) = self._frame_and_struct_buffer.peekitem()[
                1
            ][0].shape
            self._set_frame_size((width, height))

        sorted_timestamps = sorted(
            [
                frame_struct.relative_timestamp_ms
                for frame_struct in video_struct.frames
            ]
        )
        start_ts_ms = sorted_timestamps[0]
        sorted_relative_timestamps = [
            item - start_ts_ms for item in sorted_timestamps
        ]

        container = av.open(temp_filepath, mode="w")
        stream = container.add_stream("h264", rate=self._INCIDENT_VIDEO_FPS)
        stream.codec_context.time_base = Fraction(1, 1000)

        stream.width = self._frame_size[0]
        stream.height = self._frame_size[1]
        stream.pix_fmt = "yuv420p"

        for idx, _ in enumerate(sorted_relative_timestamps):
            access_ts = sorted_timestamps[idx]
            relative_timestamp_ms = sorted_relative_timestamps[idx]
            img = self._frame_and_struct_buffer[access_ts][0]
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts = int(relative_timestamp_ms)
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

        # Close the file
        container.close()

    def _write_incident_annotations(
        self, incident: Incident, video_struct: Video
    ) -> None:
        filename = f"{incident.uuid}_annotations.json"
        filepath = os.path.join(self._temp_directory, filename)

        # Convert to str since actor ids can be track_uuid which is a UUID
        # or track_id which is an int.
        actors_of_interest = set(map(str, incident.actor_ids or []))

        # Convert epoch timestamps to relative ts, such that start will be 0, same as video.
        # Probably do a deep copy before modifying.
        video_struct = copy.deepcopy(video_struct)
        start_ts_ms = min(
            frame_struct.relative_timestamp_ms
            for frame_struct in video_struct.frames
        )
        for frame_struct in video_struct.frames:
            frame_struct.relative_timestamp_ms -= start_ts_ms
            # Filter to keep only actors of interest.
            # Convert to str since track_uuid is a UUID and track_id is an int.
            frame_struct.actors = [
                actor_struct
                for actor_struct in frame_struct.actors
                if str(actor_struct.track_uuid) in actors_of_interest
                or str(actor_struct.track_id) in actors_of_interest
            ]

        with open(filepath, "w") as f:
            json.dump(video_struct.to_dict(), f)

    def _generate_video_struct(self, incident: Incident) -> Tuple[Video, List]:
        """Generates video struct for incident."""
        video = Video(uuid=None)
        start = incident.start_frame_relative_ms
        if incident.pre_start_buffer_ms is not None:
            start -= incident.pre_start_buffer_ms
        end = incident.end_frame_relative_ms
        if incident.post_end_buffer_ms is not None:
            end += incident.post_end_buffer_ms

        for key in self._frame_and_struct_buffer.irange(start, end):
            video.frames.append(self._frame_and_struct_buffer[key][1])
        sorted_timestamps = sorted(
            [
                frame_struct.relative_timestamp_ms
                for frame_struct in video.frames
            ]
        )
        return video, sorted_timestamps

    def _write_incident(self, incident: Incident) -> None:
        filename = f"{incident.uuid}_incident.json"
        filepath = os.path.join(self._temp_directory, filename)
        with open(filepath, "w") as f:
            json.dump(incident.to_dict(), f)

    def _write_mcap_log(self, incident: Incident, video: Video) -> None:
        """
        Writes the mcap log out to disk

        Args:
            incident (Incident): the incident to write
            video (Video): the video struct to serialize
        """
        filename = f"{incident.uuid}_log.mcap"
        filepath = os.path.join(self._temp_directory, filename)
        video.serialize_to_mcap(filepath)
