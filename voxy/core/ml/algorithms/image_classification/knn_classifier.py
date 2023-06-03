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
import faiss
import numpy as np

# Snake Case Naming Style:
# trunk-ignore-all(pylint/C0103)


class FaissKNeighbors:
    """
    Faiss based implementation of the KNN model. Runtime is on the order
    of O(N) for N labeled samples
    """

    def __init__(self, k=5, gpu=False):
        """
        Initializes KNN model

        Args:
            k (int, optional): The number of neighbors for KNN. Defaults to 5.
        """
        self.index = None
        self.y = None
        self.k = k
        self.gpu = gpu
        self.gpu_resource = faiss.StandardGpuResources() if gpu else None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fits KNN model based on faiss's implementation of
        L2 flat indexing

        Args:
            x (np.array): the raw x vectors (data)
            y (np.array): the raw y vectors (labels)
        """
        self.index = faiss.IndexFlatL2(x.shape[1])
        if self.gpu:
            self.index = faiss.index_cpu_to_gpu(
                self.gpu_resource, 0, self.index
            )
        self.index.add(x.astype(np.float32))
        self.y = y

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts a value based on a given data
        vector


        Args:
            x (np.array): a batched set of vectors the same size used to fit

        Returns:
            np.ndarray: the predicted label for the batched input
        """
        _, indices = self.index.search(x.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array(
            [np.argmax(np.bincount(vote)) for vote in votes]
        )
        return predictions
