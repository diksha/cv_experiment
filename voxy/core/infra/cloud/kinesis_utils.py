import fractions
import os
import time
from contextlib import ExitStack
from datetime import datetime, timedelta
from fractions import Fraction
from typing import Any, List, Optional, Tuple

import av
import boto3
import botocore.exceptions
import cv2
from loguru import logger

from core.utils.mkvtagreader import MKVReadError, MKVTagReader
from core.utils.pyav_decoder import PyavDecoder, PyavError, PyavFrame

_AWS_REGION = "us-west-2"
_KINESIS_CHUNK_DURATION_S = 30
_KINESIS_DOWNLOAD_MEDIA_TIMEOUT_S = 300


class KinesisVideoError(RuntimeError):
    pass


class KinesisVideoMediaReader:
    def __init__(
        self,
        stream_arn: Optional[str] = None,
        stream_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        fps: Optional[float] = None,
        run_once: bool = True,
        session: Optional[boto3.Session] = None,
    ) -> None:
        """This class handles reading and decoding video from a kinesis video stream

        Must provide exactly one of stream_arn or stream_name

        Args:
            stream_arn (str, optional):
                The ARN of the kinesis video stream to pull from
                Defaults to None
            stream_name (str, optional):
                The name of the kinesis video stream
                Defaults to None
            start_time (datetime, optional):
                Optional starting timestamp to pull video from. Defaults to now
            fps (float, optional):
                Causes get_frame to evenly drop frames to reduce the framerate
                of incoming video to this rate
            run_once (bool, optional):
                If set, the reader will not initiate another payload request when
                KVS closes the body of the
                previous payload
            session (boto3.Session, optional):
                Session to use for calls to kinesis video streams. Defaults to
                default boto3 session
        Raises:
            ValueError: on setting both or neither of stream_arn and stream_name
            KinesisVideoError: on failing to start kinesis video stream
        """
        super().__init__()

        if stream_arn is None and stream_name is None:
            raise ValueError("stream_arn or stream_name must be set")

        if stream_arn is not None and stream_name is not None:
            raise ValueError("stream_arn and stream_name must not both be set")

        self._run_once: bool = run_once
        self._stream_arn: Optional[str] = stream_arn
        self._stream_name: Optional[str] = stream_name
        self._fps: Optional[float] = fps
        self._start_selector = {
            "StartSelectorType": "NOW",
        }

        self._create_client = (
            boto3.client if session is None else session.client
        )

        if start_time is not None:
            self._start_selector = {
                "StartSelectorType": "PRODUCER_TIMESTAMP",
                "StartTimestamp": start_time,
            }

        self._decoder: Optional[PyavDecoder] = None
        self._tagr: Optional[MKVTagReader] = None

        try:
            # initial request start
            self._refresh_request()
        except (
            PyavError,
            MKVReadError,
            botocore.exceptions.ClientError,
        ) as err:
            raise KinesisVideoError(
                "failed to start kinesis video reader"
            ) from err

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.stop()

    def __iter__(self):
        return self

    def __next__(self) -> PyavFrame:
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame

    def _start_request(self) -> Tuple[PyavDecoder, MKVTagReader]:
        """Connect to Kinesis Video Streams and make media request

        Returns:
            Tuple[PyavDecoder, MKVTagReader]: Decoder and mkv tag reader for the stream
        """
        kvs_client = self._create_client(
            "kinesisvideo", region_name=_AWS_REGION
        )
        if self._stream_arn is not None:
            kvs_endpoint = kvs_client.get_data_endpoint(
                StreamARN=self._stream_arn,
                APIName="GET_MEDIA",
            )
        else:
            kvs_endpoint = kvs_client.get_data_endpoint(
                StreamName=self._stream_name,
                APIName="GET_MEDIA",
            )

        endpoint_url = kvs_endpoint["DataEndpoint"]
        # Get video fragment from KVS
        kvm_client = self._create_client(
            "kinesis-video-media",
            endpoint_url=endpoint_url,
            region_name=_AWS_REGION,
        )

        if self._stream_arn is not None:
            resp = kvm_client.get_media(
                StreamARN=self._stream_arn,
                StartSelector=self._start_selector,
            )
        else:
            resp = kvm_client.get_media(
                StreamName=self._stream_name,
                StartSelector=self._start_selector,
            )

        logger.debug("get_media request started")
        tagr = MKVTagReader(resp["Payload"])
        decoder = PyavDecoder(tagr, fps=self._fps, format="matroska")
        return decoder, tagr

    def _refresh_request(self) -> None:
        if (
            self._tagr is not None
            and "AWS_KINESISVIDEO_CONTINUATION_TOKEN" in self._tagr.tags()
        ):
            self._start_selector = {
                "StartSelectorType": "CONTINUATION_TOKEN",
                "ContinuationToken": self._tagr.tags()[
                    "AWS_KINESISVIDEO_CONTINUATION_TOKEN"
                ],
            }

        logger.info(
            f"get_media restarting with selector={self._start_selector}"
        )

        if self._tagr is not None:
            logger.debug(f"{self._tagr.tags()}")

        if self._decoder is not None:
            self._decoder.close()

        if self._tagr is not None:
            self._tagr.close()

        self._decoder, self._tagr = self._start_request()

    def get_height(self) -> int:
        """Gets the stream height, must be called after start()

        Returns:
            int: height in pixels
        """
        return self._decoder.get_height()

    def get_width(self) -> int:
        """Gets the stream width, must be called after start()

        Returns:
            int: width in pixels
        """
        return self._decoder.get_width()

    def get_fps(self) -> fractions.Fraction:
        """Gets the stream fps as a Fraction value

        Returns:
            fractions.Fraction: stream fps
        """
        return self._decoder.get_fps()

    def get_frame(self) -> Optional[PyavFrame]:
        """Retrieves and decodes the next frame from kinesis, must be called after start()

        Raises:
            KinesisVideoError: called before start
            RuntimeError: indicates get_frame was stuck in a retry loop

        Returns:
            Optional[PyavFrame]: None if the stream/decoder are unavailable/closed
        """
        # try to fetch a frame
        frame = None

        # time until when retry to fetch a frame
        deadline_time = time.time() + 600.0
        while frame is None:
            if time.time() > deadline_time:
                logger.debug(
                    "KinesisVideoMediaReader failure due to too many retries"
                )
                raise RuntimeError("failed after retrying for 600 seconds")

            try:
                # attempt to refresh the request if our decoder has been reset
                if self._decoder is None:
                    if self._run_once:
                        return None

                    self._refresh_request()

                # pull the next frame if we can
                if self._decoder is not None:
                    frame = self._decoder.get_frame()

                if frame is None:
                    # return as we will not be retrying
                    if self._run_once:
                        return None

                    # we still have no frame, reset the decoder
                    if self._decoder is not None:
                        self._decoder.close()
                        self._decoder = None
            except (PyavError, MKVReadError) as err:
                if self._run_once:
                    raise KinesisVideoError("video read error") from err
                logger.debug(f"retrying after error {err} reading frame")
                if self._decoder is not None:
                    self._decoder.close()
                    self._decoder = None
                # Sleep between retries.
                time.sleep(1)
            except botocore.exceptions.ClientError as err:
                raise KinesisVideoError("getMedia API failure") from err

        return frame

    def stop(self):
        """Stops and cleans up resources acquired by this reader"""
        if self._decoder is not None:
            self._decoder.close()
        if self._tagr is not None:
            self._tagr.close()


def get_archived_media_client(
    api_name: str, camera_arn: str, session: boto3.Session = None
) -> Any:
    """Gets a client for api name and camera arn

    Args:
        api_name (str): Name of the api to get endpoint from
        camera_arn (str): ARN of the kinesis video stream
        session (boto3.Session, optional):
            Session for making calls to kinesis video
            Defaults to the default boto3 session

    Returns:
        Any: Client for api
    """
    create_client = boto3.client if session is None else session.client

    kinesis_client = create_client("kinesisvideo", region_name=_AWS_REGION)
    endpoint = kinesis_client.get_data_endpoint(
        StreamARN=camera_arn, APIName=api_name
    )

    endpoint_url = endpoint["DataEndpoint"]
    return create_client(
        "kinesis-video-archived-media",
        endpoint_url=endpoint_url,
        region_name=_AWS_REGION,
    )


def download_original_video(
    camera_arn: str,
    sorted_timestamps: List,
    filepath: str,
    session: boto3.Session = None,
):
    """Downloads the original video from kinesis without re-encoding

    Note: Get clip on start timestamp, endtimestamp are not inclusive of
    those frames so we need to use list_fragments to get timestamps such that
    get clip includes start and end timestamps.
    We will store the offset of video downloaded and start timestamp in the filename
    which will be used to map the right annotations to the frames in video.

    Download media works with an assumption that no video is more than 200 fragments,
    around 25 minutes.

    Args:
        camera_arn (str): ARN of the kinesis video stream
        sorted_timestamps (List): List of timestamps in milliseconds to store
        filepath (str): Location to store the saved mp4
        session (boto3.Session, optional):
            Session for making calls to kinesis video
            Defaults to the default boto3 session

    """
    logger.info(f"Downloading original video for {filepath}")
    list_fragment_client = get_archived_media_client(
        "LIST_FRAGMENTS", camera_arn, session
    )
    get_clip_client = get_archived_media_client(
        "GET_CLIP", camera_arn, session
    )

    # Gets the list of fragments to get information about start timestamp and end timestamp.
    fragments = list_fragment_client.list_fragments(
        StreamARN=camera_arn,
        FragmentSelector={
            "FragmentSelectorType": "PRODUCER_TIMESTAMP",
            "TimestampRange": {
                "StartTimestamp": datetime.fromtimestamp(
                    sorted_timestamps[0] / 1000.0
                )
                - timedelta(seconds=_KINESIS_CHUNK_DURATION_S),
                "EndTimestamp": datetime.fromtimestamp(
                    sorted_timestamps[-1] / 1000.0
                )
                + timedelta(seconds=_KINESIS_CHUNK_DURATION_S),
            },
        },
    )["Fragments"]

    sorted_fragments = sorted(fragments, key=lambda d: d["ProducerTimestamp"])
    # Get closest start fragment
    start_timestamp_s, end_timestamp_s = None, None
    for fragment in sorted_fragments:
        if (
            fragment["ProducerTimestamp"].timestamp()
            <= sorted_timestamps[0] / 1000.0
        ):
            start_timestamp_s = fragment["ProducerTimestamp"].timestamp()
        else:
            break
    # Get closest end fragment
    for fragment in sorted_fragments:
        if (
            fragment["ProducerTimestamp"].timestamp()
            >= sorted_timestamps[-1] / 1000.0
        ):
            end_timestamp_s = (
                fragment["ProducerTimestamp"].timestamp()
                + fragment["FragmentLengthInMilliseconds"] / 1000.0
            )
            break
    # If there are more fragments to download, limit to downloading only 200 fragments
    if not end_timestamp_s and len(sorted_fragments) >= 1:
        logger.error(
            f"Total video length to download "
            f"{sorted_timestamps[-1] - sorted_timestamps[0]} "
            f"ms greater than get_clip can get."
        )
        end_timestamp_s = (
            sorted_fragments[-1]["ProducerTimestamp"].timestamp()
            + sorted_fragments[-1]["FragmentLengthInMilliseconds"] / 1000.0
        )
    # Get clip using timestamp.
    if not start_timestamp_s or not end_timestamp_s:
        logger.error(
            f"Could not download original video for timestamp "
            f"{start_timestamp_s} {end_timestamp_s} {sorted_timestamps[0]} "
            f"{sorted_timestamps[-1]} {sorted_fragments}"
        )
        return

    clip_fragment = {
        "FragmentSelectorType": "PRODUCER_TIMESTAMP",
        "TimestampRange": {
            "StartTimestamp": datetime.fromtimestamp(start_timestamp_s),
            "EndTimestamp": datetime.fromtimestamp(end_timestamp_s),
        },
    }
    payload = get_clip_client.get_clip(
        StreamARN=camera_arn,
        ClipFragmentSelector=clip_fragment,
    )["Payload"]
    offset_ms = int(sorted_timestamps[0] - (start_timestamp_s * 1000))
    filename_with_offset = (
        f"{os.path.splitext(filepath)[0]}_"
        f"{offset_ms}{os.path.splitext(filepath)[1]}"
    )
    with open(filename_with_offset, "wb") as out_file:
        for chunk in payload.iter_chunks():
            out_file.write(chunk)
        out_file.flush()
    logger.info(f"Downloaded original media to {filename_with_offset}")


def download_media(
    camera_arn: str,
    sorted_timestamps: List,
    filepath: str,
    fps: float,
    session: Optional[boto3.Session] = None,
) -> None:
    """Downloads video from kinesis and saves into an mp4 clip

    Args:
        camera_arn (str): ARN of the kinesis video stream
        sorted_timestamps (List): List of timestamps in milliseconds to store
        filepath (str): Location to store the saved mp4
        fps (float): Framerate to limit output video to
        session (boto3.Session, optional):
            session to use for fetching data from KVS. Defaults to using default boto session
    Raises:
        KinesisVideoError: indicates a parameter is invalid
        TimeoutError: indicates a timeout reading data from kinesis
    """
    if fps is None:
        raise KinesisVideoError("FPS should be defined")
    if len(sorted_timestamps) == 0:
        raise KinesisVideoError("sorted_timestamps list should not be empty")

    logger.info("Generating Clip")

    with ExitStack() as stack:
        start_datetime = datetime.fromtimestamp(
            sorted_timestamps[0] / 1000.0
        ) - timedelta(seconds=_KINESIS_CHUNK_DURATION_S)

        kinesis_media_reader = stack.enter_context(
            KinesisVideoMediaReader(
                stream_arn=camera_arn,
                start_time=start_datetime,
                fps=fps,
                session=session,
                run_once=False,
            )
        )

        # set up the encoder
        output_container = stack.enter_context(
            av.open(filepath, mode="w", format="mp4")
        )
        output_stream = output_container.add_stream("h264", rate=fps)
        output_stream.codec_context.time_base = Fraction(1, 1000)
        output_stream.pix_fmt = "yuv420p"
        output_stream.height = kinesis_media_reader.get_height()
        output_stream.width = kinesis_media_reader.get_width()

        start_ts_ms = sorted_timestamps[0]

        timeout_at = datetime.now() + timedelta(
            seconds=_KINESIS_DOWNLOAD_MEDIA_TIMEOUT_S
        )
        for pyav_frame in kinesis_media_reader:
            pts = pyav_frame.timestamp_ms()
            frame = av.VideoFrame.from_ndarray(
                pyav_frame.to_ndarray(), format="bgr24"
            )
            # we can mix pts and start_ts_ms here because codec_context.time_base is 1/1000
            frame.pts = pts - start_ts_ms

            # Include frames even if we don't have predictions for it
            # such that video can play smoothly.
            if pts >= sorted_timestamps[0]:
                for packet in output_stream.encode(frame):
                    output_container.mux(packet)
            if pts >= sorted_timestamps[-1]:
                # hit the end, time to flush and exit
                break

            if datetime.now() > timeout_at:
                raise TimeoutError("timed out downloading media from kinesis")

        # flush the stream
        for packet in output_stream.encode():
            output_container.mux(packet)


def extract_thumbnail(inputpath: str, outputpath: str) -> None:
    """Extracts a jpg thumbnail from the specified media file

    Args:
        inputpath (str): video file to read from
        outputpath (str): output path of the jpeg file
    """
    with PyavDecoder(inputpath) as decoder:
        for pyav_frame in decoder:
            frame_array = pyav_frame.to_ndarray()
            resize_dim = (
                int(
                    frame_array.shape[1] * 480.0 / float(frame_array.shape[0])
                ),
                480,
            )
            resized_frame = cv2.resize(
                frame_array, resize_dim, interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(outputpath, resized_frame)
            break
