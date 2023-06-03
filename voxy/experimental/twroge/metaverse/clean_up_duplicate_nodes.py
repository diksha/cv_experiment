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
import json

from loguru import logger
from tqdm import tqdm

from core.metaverse.migrations.lib.neomodel_context import NeoModelContext

# Trunk ignores since this is a offline testing script
# trunk-ignore-all(pylint/W9012)
# trunk-ignore-all(pylint/C0116)
# trunk-ignore-all(pylint/W9011)
# trunk-ignore-all(semgrep/python.lang.security.audit.formatted-sql-query.formatted-sql-query)
# trunk-ignore-all(semgrep/python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query)
# trunk-ignore-all(pylint/C0301)


def get_duplicate_nodes(
    database: NeoModelContext, node_type: str, batch_size: int
):
    node_ids = []
    for property_name in ["uuid", "voxel_uuid", "path"]:
        results = database.execute(
            f"""
            MATCH (v:{node_type})
            WITH v.{property_name} AS {property_name}, COLLECT(v) AS videos
            WHERE SIZE(videos) > 1
            return ID(TAIL(videos)[0]) limit {batch_size}
            """
        )
        node_ids.extend([result[0] for result in results[0]])
        logger.info(results)
    return node_ids


def delete_nodes(database: NeoModelContext, node_type: str, ids: list):
    result = database.execute(
        f"match (v:{node_type}) where ID(v) in {json.dumps(ids)} detach delete v return ID(v)"
    )
    logger.info(result)
    return result


def main():
    batch_size = 20
    with tqdm() as progress_bar:
        while True:
            with NeoModelContext() as database:
                total_node_ids = []
                for node_type in ["DataCollection", "Video"]:
                    node_ids = get_duplicate_nodes(
                        database, node_type, batch_size
                    )
                    total_node_ids.extend(node_ids)
                    if node_ids:
                        delete_nodes(database, node_type, node_ids)
                    progress_bar.update(len(node_ids))

                if not total_node_ids:
                    logger.success("Found all duplicate nodes, exiting")
                    break


if __name__ == "__main__":
    main()
