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
import typing

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


def set_nodes_unmigrated(database: NeoModelContext):
    logger.info("Setting all Video nodes to have is_migrated=False")
    result = database.execute(
        "match (v:Video) where (v.is_migrated IS NOT NULL and not v.is_migrated=True) or (v.is_migrated IS NULL)  set v.is_migrated=False"
    )
    return result


def get_unmigrated_nodes(database: NeoModelContext, count: int = 20):
    results = database.execute(
        f"match (v:Video) where v.is_migrated=False with ID(v) as id_ return id_ limit {count}"
    )
    logger.info(results)
    all_results = [result[0] for result in results[0]]
    return all_results


def clone_nodes(database: NeoModelContext, ids: typing.List[int]):
    logger.info("Clone nodes")
    results = database.execute(
        f"""
        match (v:Video) where ID(v) in {json.dumps(ids)}
        with collect(v) as videos_to_migrate
        call apoc.refactor.cloneNodes(videos_to_migrate, true) yield input, output, error
            set output.uuid=output.uuid+"-unmigrated"
            set output.voxel_uuid=output.voxel_uuid+"-unmigrated"
        return ID(output)
        """
    )
    logger.info(results)
    new_node_ids = [result[0] for result in results[0]]
    return new_node_ids


def make_video_nodes_data_collections(
    database: NeoModelContext, ids: typing.List[int]
):

    logger.info(f"Making all those data collections for node ids: {ids}")
    result = database.execute(
        f"""
        match (v:Video) where ID(v) in {json.dumps(ids)}
        with collect(v) as videos
            call apoc.refactor.rename.label("Video", "DataCollection", videos) YIELD committedOperations
         return committedOperations
        """
    )
    logger.info(result)
    return result


def set_nodes_migrated(database: NeoModelContext, ids: list):
    result = database.execute(
        f"match (v:Video) where ID(v) in {json.dumps(ids)} set v.is_migrated=True"
    )
    logger.info(result)


def strip_name_from_uuids(database: NeoModelContext, ids: list):
    result = database.execute(
        f"""
        match (d:DataCollection) where ID(d) in {json.dumps(ids)} set d.uuid=replace(d.uuid, "-unmigrated", "") set d.voxel_uuid=replace(d.voxel_uuid, "-unmigrated", "") set d.data_collection_type="VIDEO"
        """
    )
    logger.info(result)
    return result


def main():
    with NeoModelContext() as database:
        logger.info(set_nodes_unmigrated(database))
    batch_size = 2
    with tqdm() as progress_bar:
        while True:
            with NeoModelContext() as database:
                node_ids = get_unmigrated_nodes(database, count=batch_size)
                if not node_ids:
                    logger.success("Migrated all nodes, exiting")
                    break
                logger.info(node_ids)
                new_node_ids = clone_nodes(database, node_ids)

            # for some reason we can't complete a transaction this large so we have to commit two separate
            # transactions. TODO: find out why this happening
            with NeoModelContext() as database:
                make_video_nodes_data_collections(database, new_node_ids)
                set_nodes_migrated(database, node_ids)
                strip_name_from_uuids(database, new_node_ids)
            progress_bar.update(len(node_ids))


if __name__ == "__main__":
    main()
