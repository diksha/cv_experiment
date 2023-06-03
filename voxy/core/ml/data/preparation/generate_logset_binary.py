import argparse

from ml.data.preparation.generate_logset import insert_logset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logset_name", "-l", type=str, required=True)
    parser.add_argument(
        "--video_uuids", "-v", metavar="N", type=str, nargs="+", default=[]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    insert_logset(args.video_uuids, args.logset_name)


if __name__ == "__main__":
    main()
