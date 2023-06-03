import io
import json
from enum import Enum
from fractions import Fraction
from typing import List, Optional

import av
from django.contrib.auth.models import User

from core.portal.accounts.helpers import get_fullname
from core.portal.api.models.comment import Comment
from core.portal.api.models.incident import Incident
from core.structs.video import Video
from core.utils import aws_utils


def attachment_disposition(filename: str) -> str:
    return f"attachment; filename={filename}"


class StorageProvider(Enum):
    S3 = 1


def video_rendering(
    annotations_path: str,
    video_path: str,
    out_path: str,
    actor_ids: Optional[List[str]] = None,
    provider: StorageProvider = StorageProvider.S3,
) -> None:

    if provider == StorageProvider.S3:
        label_json = aws_utils.read_from_s3(annotations_path)
        bucket, path = aws_utils.separate_bucket_from_relative_path(video_path)
        video_url = aws_utils.generate_presigned_url(bucket, path)
    else:
        raise ValueError("unrecognized storage provider")

    video_struct = Video.from_dict(json.loads(label_json))

    frame_map = {}
    # filter for just highlighted actors
    for frame in video_struct.frames:
        if actor_ids:
            frame.actors = [
                actor
                for actor in frame.actors
                if str(actor.track_id) in actor_ids
                or str(actor.track_uuid) in actor_ids
            ]
        frame_map[int(frame.relative_timestamp_ms)] = frame

    in_container = av.open(video_url)
    in_stream = in_container.streams.video[0]
    in_width = in_stream.format.width
    in_height = in_stream.format.height
    in_frame_rate = int(in_stream.average_rate)
    # time_base on av input and output might be different
    # so, we need to convert it by multiply it for each frame
    # to meet our standar relative timestamp in Fraction(1, 1000)
    start_time_base = 1
    end_time_base = 1000
    time_base_ratio = in_stream.time_base * end_time_base

    temp_file = io.BytesIO()
    out_container = av.open(temp_file, mode="w", format="mp4")
    out_stream = out_container.add_stream("h264", rate=in_frame_rate)
    out_stream.codec_context.time_base = Fraction(
        start_time_base, end_time_base
    )
    out_stream.width = in_width
    out_stream.height = in_height
    out_stream.pix_fmt = "yuv420p"

    for frame in in_container.decode(in_stream):
        ts_ms = int(frame.pts * time_base_ratio)
        frame_s = frame_map.get(ts_ms)
        new_frame = av.VideoFrame.to_ndarray(frame, format="bgr24")
        if frame_s:
            new_frame = frame_s.draw(
                new_frame,
                label_type=None,
                actor_ids=actor_ids,
                draw_timestamp=False,
            )
        new_frame = av.VideoFrame.from_ndarray(new_frame, format="bgr24")
        # Set pts for new frame, otherwise it will drop to framerate 0
        new_frame.pts = ts_ms
        for packet in out_stream.encode(new_frame):
            out_container.mux(packet)

    # Flush out_stream
    for packet in out_stream.encode():
        out_container.mux(packet)

    # Close the file
    in_container.close()
    out_container.close()

    if provider == StorageProvider.S3:
        aws_utils.upload_fileobj_to_s3(
            out_path, temp_file.getvalue(), content_type="video/mp4"
        )


def log_action(
    user: User, incident: Incident, activity_type: Comment.ActivityType
) -> None:
    if activity_type == Comment.ActivityType.RESOLVE:
        text = f"resolved by {get_fullname(user)}"
    elif activity_type == Comment.ActivityType.REOPEN:
        text = f"reopened by {get_fullname(user)}"
    else:
        return

    comment = Comment(
        text=text,
        owner_id=user.pk,
        incident_id=incident.pk,
        activity_type=activity_type,
    )
    comment.save()
