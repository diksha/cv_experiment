import argparse
import sys
from datetime import datetime, timezone

from loguru import logger

from core.infra.cloud.kinesis_utils import KinesisVideoMediaReader


def main():
    parser = argparse.ArgumentParser(
        description="Pull a kinesis video stream for testing purposes"
    )
    parser.add_argument(
        "--stream-name", type=str, required=True, help="stream name to pull"
    )
    parser.add_argument(
        "--start-unix-time",
        type=lambda d: datetime.fromtimestamp(
            float(d), timezone.utc
        ).astimezone(),
        required=False,
        help="starting unix time",
    )
    parser.add_argument(
        "--end-unix-time",
        type=lambda d: datetime.fromtimestamp(
            float(d), timezone.utc
        ).astimezone(),
        required=False,
        help="end unix time",
    )
    args = parser.parse_args()

    logger.info(
        f"get media from {args.start_unix_time} to {args.end_unix_time}"
    )

    reader = KinesisVideoMediaReader(
        stream_name=args.stream_name,
        start_time=args.start_unix_time,
        run_once=False,
    )

    count = 0
    for frame in reader:
        count += 1
        logger.info(frame.timestamp_ms())

        if args.end_unix_time is not None:
            frametime = datetime.fromtimestamp(
                frame.timestamp_ms() / 1000.0, timezone.utc
            ).astimezone()
            if frametime > args.end_unix_time:
                logger.info(f"successfully decoded {count} frames")
                reader.stop()


if __name__ == "__main__":
    sys.exit(main())
