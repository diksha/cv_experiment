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


from core.metaverse.api.queries import (
    generate_data_collection_logset_from_query,
)
from core.ml.data.generation.common.logset_generator import LogsetGenerator
from core.ml.data.generation.common.registry import LogsetGeneratorRegistry
from core.structs.dataset import DataCollectionLogset


@LogsetGeneratorRegistry.register()
class DataCollectionLogsetGeneratorFromQuery(LogsetGenerator):
    """
    Data Collection Logset generator given a query
    """

    def __init__(
        self,
        query: str,
        query_name: str,
        query_variables: dict,
        logset_name: str,
    ):
        """Initializes the logset generator

        Args:
            query (str): Query that gets the list of the videos
            query_name (str): Name of the query
        """
        self.query = query
        self.query_name = query_name
        self.query_variables = query_variables
        self.logset_name = logset_name

    def generate_logset(
        self,
    ) -> DataCollectionLogset:
        """Generates a logset given metaverse query details.

        Returns:
            DataCollectionLogset: Logset generated
        """
        return generate_data_collection_logset_from_query(
            query=self.query,
            query_name=self.query_name,
            query_vars=self.query_variables,
            logset_name=self.logset_name,
        )
