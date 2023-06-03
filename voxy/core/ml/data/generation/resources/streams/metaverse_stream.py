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

import typing

from loguru import logger

from core.metaverse.metaverse import Metaverse
from core.ml.data.generation.common.registry import StreamRegistry
from core.ml.data.generation.common.stream import Stream
from core.structs.data_collection import DataCollection


@StreamRegistry.register()
class MetaverseStream(Stream):
    """
    Metaverse Stream generates a stream of video/image collection objects from a query
    and returns an iterable map
    """

    def __init__(self, query: str, query_name: str, query_variables: str):
        self.metaverse = Metaverse()
        if query_name not in query:
            logger.error(f"Could not find {query_name} in {query}")
            raise Exception("Query name must exist in graphql query")
        self.query = query
        self.query_name = query_name
        self.query_variables = query_variables

    def stream(self) -> typing.Iterable[DataCollection]:
        """
        Streams the query result and returns an iterable
        stream of data collection structs

        Raises:
            RuntimeError: if the metaverse schema query failed

        Returns:
            typing.Iterable[DataCollection]: the iterable stream of
                                 data collection structs
        """
        result = self.metaverse.schema.execute(
            self.query, variables=self.query_variables
        )
        if result.data is None:
            logger.error(result.errors)
            raise RuntimeError("Query was unsuccessful")
        logger.info(
            f"Found {len(result.data[self.query_name])} data collections"
        )
        return map(DataCollection.from_metaverse, result.data[self.query_name])
