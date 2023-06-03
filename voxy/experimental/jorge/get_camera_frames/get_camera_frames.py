import argparse
from datetime import datetime

import pytz
from google.protobuf.text_format import MessageToString

from lib.utils.fetch_camera_frames import fetch_camera_frames


def main():
    """Prints all frame structs for a camera_uuid"""
    parser = argparse.ArgumentParser(
        prog="camera_frame_fetcher",
        description="fetches camera frames and prints some basic information about them",
    )
    parser.add_argument("camera_uuid")

    time_format = "%Y-%m-%d-%H-%M"
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, time_format).replace(
            tzinfo=pytz.UTC
        ),
        default=None,
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, time_format).replace(
            tzinfo=pytz.UTC
        ),
        default=None,
    )

    args = parser.parse_args()

    for frame in fetch_camera_frames(
        args.camera_uuid, start=args.start, end=args.end
    ):
        print(MessageToString(frame))


if __name__ == "__main__":
    main()
