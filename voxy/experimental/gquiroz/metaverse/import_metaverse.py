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
from third_party.neo4j_backup.neo4j_importer import Importer

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
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Output dir",
    )
    parser.add_argument(
        "-env",
        "--metaverse_environment",
        type=str,
        default="AWS_PROD",
        help="Metaverse environment to import to",
    )
    return parser.parse_args()


def main(
    data_dir: str,
    metaverse_environment: str,
) -> None:
    """
    Extract database data
    Args:
        data_dir (str): output dir from cwd
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
        max_connection_lifetime=200,
    )
    importer = Importer(
        project_dir=data_dir,
        driver=driver,
        database="neo4j",
        input_yes=False,
    )
    logger.info(f"Importing Data from {importer.project_dir}")
    start_time = time.time()
    try:
        importer.import_data()
    finally:
        logger.info(f"Import time, {time.time() - start_time}")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.data_dir,
        args.metaverse_environment,
    )
