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

from datetime import datetime, timezone
from uuid import uuid4

from neomodel import (
    ArrayProperty,
    DateTimeProperty,
    IntegerProperty,
    JSONProperty,
    One,
    OneOrMore,
    RelationshipTo,
    StringProperty,
    StructuredNode,
    ZeroOrMore,
    ZeroOrOne,
)

# Documentation reference: https://app.clickup.com/36001183/v/dc/12ancz-10704/12ancz-15444

# trunk-ignore-all(pylint/R0903)
# trunk-ignore-all(pylint/C0115)


class Model(StructuredNode):
    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    created_timestamp = DateTimeProperty(
        default=lambda: datetime.now(timezone.utc)
    )
    name = StringProperty()

    # example: clearml link or sematic link
    run_links = ArrayProperty(StringProperty())

    # path to the location of the model in S3 or GCS
    path = StringProperty()

    # the config to be able to load the model/run metrics
    config = JSONProperty()
    dataset_ref = RelationshipTo("Dataset", "Dataset", cardinality=One)

    # model metadata
    #     these are the model metrics in a json dictionary
    #     as use cases evolve, we may promote this to a real object
    #     property or break into a metrics object
    metadata = JSONProperty()
    metrics = JSONProperty()


class Task(StructuredNode):
    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    # purpose: safety vest, door, object detector, spill
    purpose = StringProperty()

    metadata = JSONProperty()

    # optional camera reference for a task
    service_ref = RelationshipTo("Service", "Service", cardinality=OneOrMore)
    datapool_ref = RelationshipTo(
        "Datapool", "Datapool", cardinality=ZeroOrMore
    )


class Service(StructuredNode):
    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    created_timestamp = DateTimeProperty(
        default=lambda: datetime.now(timezone.utc)
    )

    # example category: example: classification, segmentation, object detection
    category = StringProperty()

    metadata = JSONProperty()
    # all models associated with this Service
    model_refs = RelationshipTo("Model", "Model", cardinality=ZeroOrMore)

    # all tagged models associated with this Service:
    best_model_ref = RelationshipTo("Model", "Model", cardinality=ZeroOrOne)
    latest_model_ref = RelationshipTo("Model", "Model", cardinality=ZeroOrOne)
    # we can add more tags here as necessary
    datapool_ref = RelationshipTo(
        "Datapool", "Datapool", cardinality=ZeroOrMore
    )

    # version should be changed as the scope of the problem changes. This would mean that
    # the architechture, or task would change in some large way to warrant a new
    # version. This also signals incompatibility with all the previously associated models
    version = IntegerProperty()


class DatasetCollection(StructuredNode):
    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    created_timestamp = DateTimeProperty(
        default=lambda: datetime.now(timezone.utc)
    )

    metadata = JSONProperty()
    # we have a series of references to the services associated with this collection
    dataset_refs = RelationshipTo("Dataset", "Dataset", cardinality=ZeroOrMore)
    task_ref = RelationshipTo("Task", "Task", cardinality=One)


class Dataset(StructuredNode):
    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    created_timestamp = DateTimeProperty(
        default=lambda: datetime.now(timezone.utc)
    )

    # this is the config used to generate this dataset
    config = JSONProperty()

    # the path to the dataset in S3/GCS.
    # WARNING: This is cached with a TTL bucket
    # so this may go away overtime
    path = StringProperty()
    # the git version used when generating the dataset
    git_version = StringProperty()
    format = StringProperty()

    data_collection_logset_ref = RelationshipTo(
        "DataCollectionLogset", "DataCollectionLogset", cardinality=ZeroOrOne
    )

    # this is used for other properties that can be useful for
    # differentiating datasets/extending them
    metadata = JSONProperty()
    version = StringProperty()


class DataCollectionLogset(StructuredNode):
    """List of data collections that will be used for training."""

    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    created_timestamp = DateTimeProperty(
        default=lambda: datetime.now(timezone.utc)
    )
    # Human readable name of the logset
    name = StringProperty()
    # List of videos belonging to the video set
    data_collection = RelationshipTo(
        "DataCollectionReference", "DataCollectionReference"
    )
    # SHA256 hash of the comma separated sorted video UUIDs
    data_collection_uuid_hash = StringProperty()
    # Version to differentiate two logsets
    version = IntegerProperty()
    # Metadata for visibility towards logset
    metadata = JSONProperty()


class DataCollectionReference(StructuredNode):
    """Light weight video reference for training."""

    # Uuid of the video
    data_collection_uuid = StringProperty()
    # Human readable name of the video/ can be voxel_uuid
    name = StringProperty()
    # Path of where the video exists
    path = StringProperty()


class Datapool(StructuredNode):
    __primarykey__ = "uuid"
    uuid = StringProperty(unique_index=True, default=uuid4)
    created_timestamp = DateTimeProperty(
        default=lambda: datetime.now(timezone.utc)
    )
    # Human readable name of the datapool
    name = StringProperty()
    url = StringProperty()
    lightly_uuid = StringProperty()
    metadata = JSONProperty()
    lightly_config = JSONProperty()
    version = IntegerProperty()

    # extra information required to trigger a datapool job
    # with lightly
    input_directory = StringProperty()
    output_directory = StringProperty()

    # Dataset type that is based on the lightly property:
    # lightly.openapi_generated.swagger_client....
    # models.dataset_type.DatasetType,
    #
    # usually `IMAGES` or `VIDEOS`
    dataset_type = StringProperty()
    ingested_data_collections = ArrayProperty(StringProperty())
