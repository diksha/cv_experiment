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
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List

import ray
from loguru import logger
from sematic.ee.ray import RayCluster, SimpleRayCluster
from tqdm import tqdm

from core.common.functional.lib.compose import Compose
from core.common.functional.registry.registry import TransformRegistry
from core.infra.sematic.shared.resources import RAY_NODE_2CPU_8GB
from core.ml.data.generation.common.registry import (
    StreamRegistry,
    WriterRegistry,
)
from core.ml.data.generation.common.stream import Stream
from core.ml.data.generation.common.writer import (
    Collector,
    LabeledDataCollection,
    Writer,
)
from core.ml.data.generation.resources.api import (  # trunk-ignore(flake8/F401,pylint/W0611)
    register_components,
)
from core.structs.dataset import DatasetFormat

# Too many instance attributes
# trunk-ignore-all(pylint/R0902)

_DEFAULT_RAY_NODES = 5

_DEFAULT_ACTIVATION_TIMEOUT_SECONDS = 60 * 60


@dataclass
class DatasetMetaData:
    """
    All required metadata for the dataset generation used
    for registration
    """

    cloud_path: str
    config: Dict[str, object]
    dataset_format: DatasetFormat
    local_path: str

    def to_dict(self) -> dict:
        """
        Converts the dataset metadata into a dictionary

        Returns:
            dict: the dict version of the dataset metadata based on the
                   asdict implementation from dataclasses
        """
        return asdict(self)


def _compose_transforms(transforms: List[str]) -> Callable:
    """
    Compose generates a composed transform from a list of transforms

    Args:
        transforms (List[str]): the list of transforms. These should all exist
                                       in the TransformRegistry

    Returns:
        Callable: the callable transform as generated from the list given. Please
            see `core.common.functional.lib.compose.Compose` for more information
    """
    transforms = [
        TransformRegistry.get_instance(**item) for item in transforms
    ]
    return Compose(transforms)


def _create_observer_transform(config: Dict[str, object]) -> Callable:
    """Composes the observer transform from the dataset config
    Args:
        config (Dict[str, object]): the configuration used to generate the pipeline
    Returns:
        Callable: Observer transform
    """
    return _compose_transforms(
        **config["dataset_generation"]["observer_transform"]
    )


def _create_data_transform(config: Dict[str, object]) -> Callable:
    """Composes the observer transform from the dataset config
    Args:
        config (Dict[str, object]): the configuration used to generate the pipeline
    Returns:
        Callable: Data transform
    """
    return _compose_transforms(
        **config["dataset_generation"]["data_transform"]
    )


def _create_label_transform(config: Dict[str, object]) -> Callable:
    """Composes the observer transform from the dataset config
    Args:
        config (Dict[str, object]): the configuration used to generate the pipeline
    Returns:
        Callable: Label transform
    """
    return _compose_transforms(
        **config["dataset_generation"]["label_transform"]
    )


def _create_test_transform(config: Dict[str, object]) -> Callable:
    """Composes the observer transform from the dataset config
    Args:
        config (Dict[str, object]): the configuration used to generate the pipeline
    Returns:
        Callable: Test transform
    """
    return _compose_transforms(
        **config["dataset_generation"]["is_test_transform"]
    )


def run_pipeline(config: Dict[str, object]) -> DatasetMetaData:
    """
    run_pipeline brings all the streams, writers and transforms together to
    generate a dataset. The data transform and label transforms are designed
    to be generated using the `Compose([Transform ...])` framework and the
    writer is designed to work directly with the dataset.

    Components:
        source (Stream): the source stream to see some examples, see:
            `MetaverseStream`
        reader (Stream): the reader stream takes in an item from the
            source stream and produces "data" and "labels"
        ray_node_config (RayNodeConfig): compute characteristics of a
            single Ray head/worker
        ray_cluster_config (RayClusterConfig): representation of the compute
            resources the cluster will use
        data_transform (Callable): the data transforms are applied
            directly on the raw data and the observed label
        label_transform (Callable): the label transforms are
            applied directly on the raw data and the observed label, produces
            a label for consumption by the writer
        observer (Callable): the observer is designed to return an
            iterable item to be labeled. In the case of classification it is all
            the items in the frame that fit the criteria of the classifier
            (are they occluded/are they the right actor type, etc.). This is
            intended to return an iterable
        is_test (Callable): the function to return if the item from
            the source stream is test or train
        writer (Writer): the writing object that consumes the label, data and the
            test/train label

    Args:
        config (Dict[str, object]): the configuration used to generate the pipeline
            (used in registration)

    Returns:
        DatasetMetaData: all required dataset metadata used in registering the
            generated dataset
    """
    logger.info("Generating dataset")
    source: Stream = StreamRegistry.get_instance(
        **config["dataset_generation"]["source"]
    )
    writer: Writer = WriterRegistry.get_instance(
        **config["dataset_generation"]["writer"]
    )
    ray_cluster_config = SimpleRayCluster(
        node_config=RAY_NODE_2CPU_8GB,
        n_nodes=config["dataset_generation"]
        .get("ray", {})
        .get("n_nodes", _DEFAULT_RAY_NODES),
    )
    with RayCluster(
        config=ray_cluster_config,
        activation_timeout_seconds=_DEFAULT_ACTIVATION_TIMEOUT_SECONDS,
    ):
        labeled_data_refs = [
            _generate_data_for_item.remote(
                item=item,
                collector=writer.create_collector(),
                config=config,
            )
            for item in tqdm(source.stream())
        ]
        labeled_data_collections: List[LabeledDataCollection] = ray.get(
            labeled_data_refs
        )

    metadata = writer.write_dataset(labeled_data_collections)
    return DatasetMetaData(
        cloud_path=metadata.cloud_path,
        config=config,
        dataset_format=writer.format(),
        local_path=metadata.local_directory,
    )


@ray.remote(max_retries=3)
def _generate_data_for_item(
    item: object,
    collector: Collector,
    config: Dict[str, object],
) -> LabeledDataCollection:
    reader = StreamRegistry.get_class(config["dataset_generation"]["reader"])
    observer_transform = _create_observer_transform(config)
    data_transform = _create_data_transform(config)
    label_transform = _create_label_transform(config)
    is_test_transform = _create_test_transform(config)

    for stream_data in reader(item).stream():
        if stream_data.datum is None or stream_data.label is None:
            continue
        for observation in observer_transform(stream_data.label):
            transformed_data = data_transform(stream_data.datum, observation)
            transformed_label = label_transform(stream_data.datum, observation)
            is_test = is_test_transform(item)
            collector.collect_and_upload_data(
                transformed_data,
                transformed_label,
                is_test,
                stream_data.metadata,
            )
    return collector.dump()
