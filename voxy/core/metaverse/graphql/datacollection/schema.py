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

import graphene
from neomodel import db

from core.metaverse.models.datacollection import Actor, DataCollection


class VersionedViolationsSchema(graphene.ObjectType):
    version = graphene.String()
    violations = graphene.List(graphene.String)


class ActorsSchema(graphene.ObjectType):
    category = graphene.String()
    occluded = graphene.Boolean()
    occluded_degree = graphene.String()
    truncated = graphene.Boolean()
    human_operating = graphene.Boolean()
    forklift = graphene.Boolean()
    loaded = graphene.Boolean()
    forks_raised = graphene.Boolean()
    operating_pit = graphene.Boolean()
    operating_object = graphene.String()
    door_state = graphene.String()
    door_type = graphene.String()
    track_id = graphene.Int()
    track_uuid = graphene.String()
    polygon = graphene.JSONString()
    pose = graphene.String()
    activity = graphene.JSONString()
    is_wearing_safety_vest_v2 = graphene.Boolean()
    is_wearing_hard_hat = graphene.Boolean()
    motion_detection_zone_state = graphene.String()
    is_carrying_object = graphene.Boolean()
    head_covering_type = graphene.String()


class FrameSchema(graphene.ObjectType):
    uuid = graphene.String()
    frame_number = graphene.String()
    frame_width = graphene.Int()
    frame_height = graphene.Int()
    relative_timestamp_ms = graphene.Float()
    actors_ref = graphene.List(ActorsSchema)
    relative_image_path = graphene.String()


class LabelMetadataSchema(graphene.ObjectType):
    """Schema for querying label metadata node

    Args:
        graphene: graphql output object type
    """

    uuid = graphene.String()
    source = graphene.String()
    project_name = graphene.String()
    taxonomy = graphene.String()
    taxonomy_version = graphene.List(graphene.String)


class DataCollectionSchema(graphene.ObjectType):
    # TODO: see what else needs to be added in here
    data_collection_type = graphene.String()
    uuid = graphene.String()
    name = graphene.String()
    path = graphene.String()
    voxel_uuid = graphene.String()
    frame_ref = graphene.List(FrameSchema)
    is_test = graphene.Boolean()
    is_deprecated = graphene.Boolean()
    violation_ref = graphene.List(VersionedViolationsSchema)
    label_metadata_ref = graphene.Field(LabelMetadataSchema)


class ActorQueries(graphene.ObjectType):
    actor_from_data_collection_slice = graphene.List(
        ActorsSchema,
        track_uuid=graphene.String(),
        data_collection_slice=graphene.String(required=True),
    )

    actors_from_data_collection_slice = graphene.List(
        ActorsSchema,
        track_uuids=graphene.List(graphene.String),
        data_collection_slice=graphene.String(),
    )

    def resolve_actor_from_data_collection_slice(
        self, info: graphene.ResolveInfo, *args, **kwargs
    ) -> "ActorQueries":
        """
        Simple resolver for getting an actor from
        a data collection slice

        Args:
            info(graphene.ResolveInfo): the resolver information
            *args: the input arguments
            **kwargs: the extra keyword arguments

        Raises:
            ValueError: when the data collection slice is not 'test'
                         or 'train'

        Returns:
            (ActorQueries): the resulting actor query result
        """
        data_collection_slice = kwargs["data_collection_slice"].lower()
        if data_collection_slice not in ("train", "test"):
            raise ValueError("DataCollection slice must be train or test")
        is_test = "true" if data_collection_slice == "test" else "false"
        query = (
            f'MATCH (a: Actor{{track_uuid: "{kwargs.get("track_uuid")}"}})--(f:Frame)--'
            f"(v:DataCollection{{is_test: {is_test}}}) RETURN distinct a.uuid"
        )
        results, _ = db.cypher_query(query)
        uuids = [uuid[0] for uuid in results]
        return Actor().nodes.filter(uuid__in=uuids)

    def resolve_actors_from_data_collection_slice(
        self, info, *args, **kwargs
    ) -> "ActorQueries":
        """
        Simple resolver for getting actors from
        a data collection slice

        Args:
            info(graphene.ResolveInfo): the resolver information
            *args: the input arguments
            **kwargs: the extra keyword arguments

        Raises:
            ValueError: when the data collection slice is not 'test'
                         or 'train'

        Returns:
            (ActorQueries): the resulting actor query result
        """
        data_collection_slice = kwargs["data_collection_slice"].lower()
        if data_collection_slice not in ("train", "test"):
            raise ValueError("DataCollection slice must be train or test")
        is_test = "true" if data_collection_slice == "test" else "false"
        query = (
            f"MATCH (a: Actor)--(f:Frame)--(v:DataCollection{{is_test: {is_test}}}) WHERE ANY "
            f'(track_uuid IN a.track_uuid WHERE track_uuid IN {kwargs.get("track_uuids")}) '
            "RETURN distinct a.uuid"
        )
        results, _ = db.cypher_query(query)
        uuids = [uuid[0] for uuid in results]
        return Actor().nodes.filter(uuid__in=uuids)


class DataCollectionQueries(graphene.ObjectType):
    data_collection = graphene.List(
        DataCollectionSchema,
        uuid=graphene.String(required=True),
    )

    data_collection_from_voxel_uuid = graphene.List(
        DataCollectionSchema,
        voxel_uuid=graphene.String(required=True),
    )

    data_collection_path_contains = graphene.List(
        DataCollectionSchema,
        path=graphene.String(required=True),
    )

    data_collection_contains_actor_categories = graphene.List(
        DataCollectionSchema,
        categories=graphene.List(graphene.NonNull(graphene.String)),
        data_collection_types=graphene.List(graphene.String),
    )

    data_collection_with_actor_category_from_site = graphene.List(
        DataCollectionSchema,
        category=graphene.String(required=True),
        organization=graphene.String(required=True),
        location=graphene.String(required=True),
        zone=graphene.String(required=True),
    )

    data_collection_with_actor_category_from_camera_uuids = graphene.List(
        DataCollectionSchema,
        category=graphene.String(required=True),
        camera_uuids=graphene.List(graphene.String),
    )

    data_collection_from_data_collection_logset = graphene.List(
        DataCollectionSchema,
        data_collection_logset_uuid=graphene.String(),
    )

    def resolve_data_collection(
        self, info: graphene.ResolveInfo, *args, **kwargs
    ) -> DataCollection:  # pylint: disable=unused-argument
        """
        Simple resolver to resolve a data collection from a uuid

        Args:
            info (graphene.ResolveInfo): the graphene resolver info
            *args: extra input arguments
            **kwargs: extra key word arguments

        Returns:
            DataCollection: the data collection filtered by uuid
        """
        return DataCollection().nodes.filter(uuid=kwargs["uuid"])

    def resolve_data_collection_from_voxel_uuid(
        self, info, *args, **kwargs
    ) -> DataCollection:  # pylint: disable=unused-argument
        """This is a GraphQL resolver for data_collection_from_voxel_uuid

        Args:
            info (info): resolver info
            args (str): args as defined by the schema
            kwargs (int): args as defined by the schema

        Returns:
            DataCollection: all data_collections with the voxel_uuid
        """
        return DataCollection().nodes.filter(
            voxel_uuid__contains=kwargs["voxel_uuid"]
        )

    def resolve_data_collection_path_contains(
        self, info, *args, **kwargs
    ) -> DataCollection:  # pylint: disable=unused-argument
        """
        Simple resolver to resolve a data collection from a path

        Args:
            info (graphene.ResolveInfo): the graphene resolver info
            *args: extra input arguments
            **kwargs: extra key word arguments

        Returns:
            DataCollection: the data collection filtered by path
        """
        return DataCollection().nodes.filter(path__contains=kwargs["path"])

    def resolve_data_collection_contains_actor_categories(
        self, info, categories: graphene.List, *args, **kwargs
    ) -> DataCollection:  # pylint: disable=unused-argument
        """
        Simple resolver to resolve a data collection from an actor category

        Args:
            info (graphene.ResolveInfo): the graphene resolver info
            categories (graphene.List): actor category list to filter for
            *args: extra input arguments
            **kwargs: extra key word arguments

        Returns:
            DataCollection: the data collection filtered by actors in the frames
        """
        if data_collection_types := kwargs.get("data_collection_types"):
            data_collection_filter = (
                f"where v.data_collection_type in {data_collection_types} "
                f"and a.category in {categories}"
            )
        else:
            data_collection_filter = f"where a.category in {categories}"
        query = (
            f"MATCH (v: DataCollection)--(f:Frame)--"
            f"(a: Actor) {data_collection_filter} return distinct v.uuid"
        )
        results, _ = db.cypher_query(query)
        uuids = [uuid[0] for uuid in results]
        return DataCollection().nodes.filter(uuid__in=uuids)

    def resolve_data_collection_with_actor_category_from_site(
        self, info, *args, **kwargs
    ) -> DataCollection:  # pylint: disable=unused-argument
        """
        Simple resolver to resolve a data collection from an actor category
        and a particular location and organization

        Args:
            info (graphene.ResolveInfo): the graphene resolver info
            *args: extra input arguments
            **kwargs: extra key word arguments

        Returns:
            DataCollection: the data collection filtered by actors in the frames
                            and the original source of the location
        """
        query = (
            f'MATCH (n: Actor{{category: "{kwargs["category"]}"}})--'
            f'(f:Frame)--(vi:DataCollection)--(C:Camera{{organization: "{kwargs["organization"]}", '
            f'location: "{kwargs["location"]}", zone: "{kwargs["zone"]}"}}) '
            "return distinct vi.uuid"
        )
        results, _ = db.cypher_query(query)
        uuids = [uuid[0] for uuid in results]
        return DataCollection().nodes.filter(uuid__in=uuids)

    def resolve_data_collection_with_actor_category_from_camera_uuids(
        self, _, *__, **kwargs
    ) -> list:
        """
        Queries the data_collections with the given actor category
        that are related to the camera with the given camera uuid

        Args:
            kwargs:
                category: the actor category to query for (e.g. DOOR)
                camera_uuids: the camera uuids to query for
            *__: unused

        Returns:
            list: the data_collection objects
        """
        query = (
            f'MATCH (n: Actor{{category: "{kwargs["category"]}"}})'
            "--(f:Frame)--(vi:DataCollection)--"
            f'(C:Camera) where C.uuid in {kwargs["camera_uuids"]} return distinct vi.uuid'
        )
        results, _ = db.cypher_query(query)
        uuids = [uuid[0] for uuid in results]
        return DataCollection().nodes.filter(uuid__in=uuids)

    def resolve_data_collection_from_data_collection_logset(
        self, _, *__, **kwargs
    ) -> list:
        """
        Queries the data_collections with the given actor category
        that are related to the camera with the given camera uuid

        Args:
            kwargs:
                uuid: the data_collection logset uuid
            *__: unused

        Returns:
            list: the data_collection objects
        """
        query = (
            "MATCH (v:DataCollectionLogset"
            f"{{uuid: '{kwargs.get('data_collection_logset_uuid')}'}}) "
            "-- (data_collection:DataCollectionReference)"
            " return distinct data_collection.data_collection_uuid"
        )
        results, _ = db.cypher_query(query)
        uuids = [uuid[0] for uuid in results]
        return DataCollection().nodes.filter(uuid__in=uuids)


class DataCollectionTestQueries(graphene.ObjectType):
    """
    These are test queries designed to help speedup developer
    workflows (like downsampling full queries)

    A lot of these will mirror the full queries in DataCollectionQueries
    """

    data_collection_test_sample_with_actor_category_from_camera_uuids = (
        graphene.List(
            DataCollectionSchema,
            category=graphene.String(),
            camera_uuids=graphene.List(graphene.String),
            count=graphene.Int(),
        )
    )
    data_collection_test_sample_contains_actor_categories = graphene.List(
        DataCollectionSchema,
        categories=graphene.List(graphene.NonNull(graphene.String)),
        data_collection_types=graphene.List(graphene.String),
        count=graphene.Int(),
    )

    def resolve_data_collection_test_sample_with_actor_category_from_camera_uuids(
        self, _, *__, **kwargs
    ) -> list:
        """
        Queries the data_collections with the given actor category
        that are related to the camera with the given camera uuid

        This is a "test" query for use cases where you do not want to
        use all results from metaverse (as when generating/testing
        dataset generation or model training). The sample/count
        that is given is stochastic so this should be used with care.

        Args:
            kwargs:
                category: the actor category to query for (e.g. DOOR)
                camera_uuids: the camera uuids to query for
                count: the max of data_collections to grab. Should be even
            *__: unused

        Returns:
            list: the data_collection objects
        """
        half_count = kwargs["count"] // 2
        test_query = (
            f'MATCH (n: Actor{{category: "{kwargs["category"]}"}})'
            f"--(f:Frame)--(vi:DataCollection {{is_test: True}})--"
            f'(C:Camera) where C.uuid in {kwargs["camera_uuids"]} '
            f"return distinct vi.uuid limit {half_count}"
        )
        results, _ = db.cypher_query(test_query)
        test_uuids = [uuid[0] for uuid in results]
        train_query = (
            f'MATCH (n: Actor{{category: "{kwargs["category"]}"}})'
            f"--(f:Frame)--(vi:DataCollection {{is_test: False}})--"
            f'(C:Camera) where C.uuid in {kwargs["camera_uuids"]}'
            f" return distinct vi.uuid limit {half_count}"
        )
        results, _ = db.cypher_query(train_query)
        train_uuids = [uuid[0] for uuid in results]
        uuids = test_uuids + train_uuids

        return DataCollection().nodes.filter(uuid__in=uuids)

    def resolve_data_collection_test_sample_contains_actor_categories(
        self,
        info: graphene.ResolveInfo,
        categories: graphene.List,
        count: graphene.Int,
        *args,
        **kwargs,
    ) -> list:
        """
        Queries the data_collections with the given actor categories.
        Optionally restricts with data_colletion_type

        This is a "test" query for use cases where you do not want to
        use all results from metaverse (as when generating/testing
        dataset generation or model training). The sample/count
        that is given is stochastic so this should be used with care.

        Args:
            info(graphene.ResolveInfo): the resolver information
            categories(graphene.List): the actor category to query for
            count(graphene.Int): the max of data_collections to grab (even)
            kwargs:
                data_collection_types: optional constraint to pull specific
                    data collection types
            *args: unused

        Returns:
            list: the data_collection objects
        """
        half_count = count // 2
        if data_collection_types := kwargs.get("data_collection_types"):
            data_collection_filter = (
                f"where v.data_collection_type in {data_collection_types} "
                f"and a.category in {categories} return distinct v.uuid "
                f"limit {half_count}"
            )
        else:
            data_collection_filter = f"where a.category in {categories}"
        test_query = (
            f"MATCH (v: DataCollection {{is_test: True}})--(f:Frame)--"
            f"(a: Actor) {data_collection_filter}"
        )
        results, _ = db.cypher_query(test_query)
        test_uuids = [uuid[0] for uuid in results]
        train_query = (
            f"MATCH (v: DataCollection {{is_test: False}})--(f:Frame)--"
            f"(a: Actor) {data_collection_filter}"
        )
        results, _ = db.cypher_query(train_query)
        train_uuids = [uuid[0] for uuid in results]
        uuids = test_uuids + train_uuids
        return DataCollection().nodes.filter(uuid__in=uuids)
