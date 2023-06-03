"""
Import pipeline and resolve
"""
import argparse
import sys

from loguru import logger
from sematic import CloudResolver, SilentResolver, has_container_image

from core.infra.sematic.shared.utils import block_until_done
from experimental.nasha.sematic.spills_pipeline.pipeline import pipeline


# trunk-ignore-all(pylint/E1101)
def main(
    spill_config_path: str,
    lightly_config_path: str,
    silent: bool = False,
) -> int:
    """Function to run FP dataset collection

    Args:
        Args:
        spill_config_path(str): A path to a yaml containing cameras,
                                sites and zones for data collection
        lightly_config_path(str): A path to a yaml containing lightly downsampling for spills
        silent (bool, optional): silet parameter for sematic resolver. Defaults to False.

    Returns:
        int: return value
    """

    future = pipeline(
        spill_config_path=spill_config_path,
        lightly_config_path=lightly_config_path,
    )

    if has_container_image():
        resolver = CloudResolver(max_parallelism=10)
        logger.info(f"Launching {future.id} in the cloud")
    elif silent:
        resolver = SilentResolver()
        logger.info(f"Launching {future.id} locally and silently")
    else:
        resolver = None
        logger.info(f"Launching {future.id} locally")

    future.resolve(resolver)
    block_until_done(future.id)
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spill_config_path",
        type=str,
        required=True,
        help=(
            "A path to a yaml containing cameras, sites and zones for data collection"
        ),
    )
    parser.add_argument(
        "--lightly_config_path",
        type=str,
        required=True,
        help=("A path to a yaml containing lightly downsampling for spills"),
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        default=False,
        help="Use sematic silent resolver",
    )

    args, _ = parser.parse_known_args()
    sys.exit(
        main(
            args.spill_config_path,
            args.lightly_config_path,
            silent=args.silent,
        )
    )
