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
from types import TracebackType

from loguru import logger
from neomodel import db

from core.metaverse.metaverse import Metaverse


class NeoModelContext:
    """
    Context to operate with the neo model database

    Automatically catches any exceptions found within the scope and rollsback the Db
    to an earlier version if any exception was raised
    """

    def __init__(self):
        self.metaverse = Metaverse()

    def __enter__(self) -> "NeoModelContext":
        """
        Enters the neomodel context. Begins a database transaction

        Returns:
            NeoModelContext: the current neomodel database instance
        """
        logger.info("Starting database transaction")
        db.begin()
        return self

    def execute(self, query: str) -> object:
        """
        Executes the query. To be used within the database
        context

        Args:
            query (str): the current cypher query to execute in metaverse

        Returns:
            object: the query response from the neomodel database
        """
        logger.debug(f"Executing query: \n{query}")
        return db.cypher_query(query)

    def __exit__(
        self,
        exception: type,
        exception_values: Exception,
        traceback: TracebackType,
    ):
        """
        Exits the database context

        Args:
            exception (type): the exception (if any) that was raised in the scope
            exception_values (Exception): the specific exception type
            traceback (traceback): the current traceback of the exception
        """
        if exception is None:
            logger.success("Successfully completed transaction. Committing")
            db.commit()
        else:
            logger.exception(
                "Error encountered during execution, rolling back"
            )
            db.rollback()
            logger.exception("traceback")
