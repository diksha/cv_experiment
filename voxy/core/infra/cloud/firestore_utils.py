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
from typing import List

from google.api_core.exceptions import NotFound
from google.cloud import firestore


def read_from_output_store(collection: str, uuid: str, key: str) -> List:
    """
    Reads value of key from google cloud firestore table for given uuid.

    Args:
        collection (str): the collection to read from
        uuid (str): the uuid of the item in the collection
        key (str): the key of document

    Raises:
        RuntimeError: if the document is invalid or the key is invalid

    Returns:
        List: the list of items in the store
    """
    db = firestore.Client()
    doc_ref = db.collection(collection).document(uuid)
    if (
        not doc_ref.get()
        or not doc_ref.get().to_dict().get("output")
        or doc_ref.get().to_dict().get("output").get(key) is None
    ):
        raise RuntimeError(
            f"Output with key {key} expected for {collection} with id {uuid}"
        )
    return doc_ref.get().to_dict().get("output").get(key)


def append_to_output_store(
    collection: str, uuid: str, key: str, value: List
) -> None:
    """
    Appends value to key in google cloud firestore table for given uuid atomically.

    Args:
        collection (str): the collection to append to
        uuid (str): the uuid of the document in the collection
        key (str): the key of the item
        value (List): the list of the
    """

    db = firestore.Client()
    doc_ref = db.collection(collection).document(uuid)
    doc_ref.update({f"output.{key}": firestore.ArrayUnion(value)})


def write_to_output_store(
    collection: str, uuid: str, key: str, value: List
) -> None:
    """
    Write value to key in google cloud firestore table for given uuid atomically.

    Args:
        collection (str): the collection to write to
        uuid (str): the uuid inside the collection
        key (str): the key of the item to store
        value (List): the value for the output store
    """
    db = firestore.Client()
    doc_ref = db.collection(collection).document(uuid)
    try:
        doc_ref.update({f"output.{key}": value})
    except NotFound:
        doc_ref.set({"output": {key: value}})
