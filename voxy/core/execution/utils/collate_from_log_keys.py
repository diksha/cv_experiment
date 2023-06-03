import argparse
import typing

from core.execution.utils.collate_logs import AggregatedResult, collate_logs
from core.utils.aws_utils import glob_from_bucket

#     usage: collate_from_log_keys.py [-h] --log_keys LOG_KEYS [LOG_KEYS ...]
#
# Find matching paths in an S3 bucket and collate all the logs. Example Usage:
# ./bazel run //core/execution/utils:collate_from_log_keys --
#                         --log_keys tim/yolo_main_comparison tim/yolo_gpu_io
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --log_keys LOG_KEYS [LOG_KEYS ...]
#                         List of log keys to compare
#


def get_matching_paths(
    keys: typing.List[str],
    prefix="logs",
    bucket_name="voxel-temp",
) -> typing.List[AggregatedResult]:
    """
    Gets matching paths for logs in s3 bucket

    Args:
        keys (typing.List[str]): the keys
        prefix (str): the prefix to prepend
        bucket_name (str, optional): the bucket name. Defaults to "voxel-temp".

    Yields:
        typing.List[AggregatedResult]: the matching paths
    """

    prefix_paths = {}
    common_paths = None

    # Iterate over the prefixes
    for key in keys:
        # Get the objects in the current prefix
        path = "/".join([prefix, key])
        prefix_paths[key] = glob_from_bucket(
            bucket_name, path, extensions=("")
        )
        stripped = [p.replace(path, "").lstrip("/") for p in prefix_paths[key]]
        if common_paths is None:
            common_paths = set(stripped)
        else:
            common_paths = set(stripped).intersection(common_paths)
        prefix_paths[key] = [
            f"s3://{bucket_name}/{path}" for path in prefix_paths[key]
        ]

    def name_to_prefix(name: str) -> str:
        """
        Converts the name to the fully qualified s3 path

        Args:
            name (str): the name

        Returns:
            str: the fully qualified s3 path
        """
        return f"s3://{bucket_name}/{prefix}/{name}"

    def strip_log_name(item: str) -> str:
        """
        Strips the log name from the path

        Args:
            item (str): the item to strip

        Returns:
            str: stripped name
        """
        return "/".join(item.split("/")[:-1])

    for common_path in common_paths:
        yield AggregatedResult(
            names=keys,
            s3_paths=list(
                "/".join([name_to_prefix(key), common_path]) for key in keys
            ),
            common_path=strip_log_name(common_path),
        )


def get_args() -> argparse.Namespace:
    """
    Gets input args

    Returns:
        argparse.Namespace: the input args
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Find matching paths in an S3 bucket and collate all the logs. Example Usage:"
        "./bazel run //core/execution/utils:collate_from_log_keys -- "
        "--log_keys tim/yolo_main_comparison  tim/yolo_gpu_io"
    )
    parser.add_argument(
        "--log_keys",
        nargs="+",
        required=True,
        help="List of log keys to compare",
    )
    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = get_args()
    matching_paths = get_matching_paths(args.log_keys)
    for paths in matching_paths:
        # collate_logs(paths)
        collate_logs(paths)
