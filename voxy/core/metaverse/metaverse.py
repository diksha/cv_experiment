#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import json
import os
import threading
from typing import Optional

from neomodel import config, db

from core.metaverse.graphql.schemas import schema
from core.utils.aws_utils import get_secret_from_aws_secret_manager

database_details = {
    "PROD": {
        "host": "metaverse-neo4j.private.voxelplatform.com:7687",
        "arn": (
            "arn:aws:secretsmanager:us-west-2:203670452561:"
            "secret:PROD_NEO4J_METAVERSE_PASSWORD-kAuS4W"
        ),
        "version": "aws",
        "protocol": "neo4j+s",
    },
    "INTERNAL": {
        "host": "metaverse-internal-neo4j.private.voxelplatform.com:7688",
        "arn": (
            "arn:aws:secretsmanager:us-west-2:203670452561:"
            "secret:INTERNAL_NEO4J_METAVERSE_PASSWORD-3B94Wg"
        ),
        "version": "aws",
        "protocol": "neo4j+s",
    },
    "LOCALHOST": {
        "host": "localhost:7687",
        "arn": (
            "arn:aws:secretsmanager:us-west-2:203670452561:"
            "secret:LOCALHOST_NEO4J_METAVERSE_PASSWORD-gZ560i"
        ),
        "version": "localhost",
        "protocol": "neo4j",
    },
}


class Metaverse:
    """
    Access to Metaverse database.

    Right now only sets defaults and enforces consistent environment.
    In future this singleton may be able to avoid global variables in neomodel
    """

    __instance_lock = threading.Lock()
    __instance: Optional["Metaverse"] = None

    @staticmethod
    def __get_environment() -> str:
        """Get metaverse environment defined by environment variables.

        Returns:
            str: name of metaverse environment
        """
        return os.getenv("METAVERSE_ENVIRONMENT")

    def __new__(cls, environment: Optional[str] = None) -> "Metaverse":
        """
        Retrieve instance of Metaverse class.

        Arguments:
            environment (Optional[str]): environment to use (otherwise retrieved from env. var)
        Raises:
            Exception: tried to instantiate singleton with different environment
        Returns:
            Metaverse: instance of Metaverse client
        """
        # It's not easy to get around calling this in both __new__ and __init__
        if environment is None:
            environment = cls.__get_environment()
            if environment is None:
                raise Exception(
                    "METAVERSE_ENVIRONMENT environment variable required!"
                )

        if not cls.__instance:
            with cls.__instance_lock:
                # double-checked locking does work thanks to GIL
                # https://mail.python.org/pipermail/python-dev/2005-October/057062.html
                if not cls.__instance:
                    cls.__instance = object.__new__(cls)
        elif environment != cls.__instance.environment:
            raise Exception(
                "Metaverse: unsupported use of multiple metaverse environments in one process"
            )
        return cls.__instance

    def __init__(self, environment: Optional[str] = None):
        """Create Metaverse client

        Args:
            environment (Optional[str]): environment to use
        """
        if environment is None:
            environment = self.__get_environment()

        self.environment = environment
        config.DATABASE_URL = self.database_url(environment)
        config.MAX_CONNECTION_LIFETIME = 200

    @staticmethod
    def database_password(environment) -> str:
        """Get database password `environment`

        Args:
            environment (str): Environment for database

        Returns:
            str: Metaverse database password for environment
        """
        database_detail = database_details[environment]
        return json.loads(
            get_secret_from_aws_secret_manager(database_detail["arn"])
        )[database_detail["version"]]

    @staticmethod
    def database_uri(environment) -> str:
        """Get database host `environment`

        Args:
            environment (str): Environment for database

        Returns:
            str: Metaverse database host for environment
        """
        database_detail = database_details[environment]
        return f"neo4j://{database_detail['host']}"

    @staticmethod
    def database_url(environment) -> str:
        """Get database URL for `environment`

        Args:
            environment (str): Environment for database

        Returns:
            str: Metaverse database URL for environment
        """
        database_detail = database_details[environment]
        password = json.loads(
            get_secret_from_aws_secret_manager(database_detail["arn"])
        )[database_detail["version"]]
        return f"{database_detail['protocol']}://neo4j:{password}@{database_detail['host']}"

    @staticmethod
    def is_database_encrypted(environment) -> bool:
        """Is db encrypted based on protocol

        Args:
            environment (str): Environment for database

        Returns:
            bool: whether db is encrypted
        """
        protocol = database_details[environment]["protocol"]
        return "+s" in protocol

    @property
    def schema(self):
        return schema

    def close(self):
        """Close the connection to neo4j driver and clear the singleton.

        Note: Explicitly call close at end of script. In future
        releases of the driver, auto close will be removed.
        (https://github.com/neo4j/neo4j-python-driver/blob/
        f43e95cf29b4a901ccc51d29eca4cc9b1db9749f/neo4j/_sync/driver.py#L401)
        """
        if db.driver:
            db.driver.close()
