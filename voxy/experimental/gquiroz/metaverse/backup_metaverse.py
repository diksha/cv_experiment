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

import argparse
import time

from loguru import logger
from neo4j import GraphDatabase

from core.metaverse.metaverse import Metaverse
from third_party.neo4j_backup.neo4j_extractor import Extractor

METAVERSE_USER = "neo4j"
TRUST = "TRUST_ALL_CERTIFICATES"


def parse_args() -> argparse.Namespace:
    """
    Argument Parser
    Returns:
        argparse.Namespace: cl args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output dir",
    )
    parser.add_argument(
        "-env",
        "--metaverse_environment",
        type=str,
        default="INTERNAL",
        help="Metaverse environment to backup",
    )
    return parser.parse_args()


def main(
    output_dir: str,
    metaverse_environment: str,
) -> None:
    """
    Extract database data
    Args:
        output_dir (str): output dir from cwd
        metaverse_environment (str): metaverse_env
    """
    uri = Metaverse.database_uri(metaverse_environment)
    password = Metaverse.database_password(metaverse_environment)
    encrypted = Metaverse.is_database_encrypted(metaverse_environment)
    driver = GraphDatabase.driver(
        uri,
        auth=(METAVERSE_USER, password),
        encrypted=encrypted,
        trust=TRUST,
        max_connection_lifetime=100,
    )
    extractor = Extractor(
        project_dir=output_dir,
        driver=driver,
        database="neo4j",
        input_yes=False,
        compress=True,
    )
    logger.info(f"Extracting Data to {extractor.project_dir}")
    start_time = time.time()
    extractor.extract_data()
    logger.info(f"Extraction time, {time.time() - start_time}")


if __name__ == "__main__":
    args = parse_args()
    main(args.output_dir, args.metaverse_environment)
