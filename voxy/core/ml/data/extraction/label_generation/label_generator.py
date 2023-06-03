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

from abc import ABC, abstractmethod

from core.structs.frame import Frame


class LabelGenerator(ABC):
    @abstractmethod
    def process_frame(
        self,
        frame_timestamp_ms: int,
        frame: Frame,
    ) -> None:
        """Process a single frame with corresponding frame labels

        Args:
            frame_timestamp_ms: timestamp of frame in ms
            frame: frame object from consumable labels map

        Returns:
            None
        """
        raise NotImplementedError(
            "LabelGenerator must implement process_frame method"
        )

    @abstractmethod
    def publish_labels(
        self,
        save_dir: str,
    ) -> None:
        """Publishes labels using client storage publisher

        Args:
            save_dir: save dir for labels

        Returns:
            None
        """
        raise NotImplementedError(
            "LabelGenerator must implement publish_labels method"
        )

    @property
    @abstractmethod
    def label_type(self) -> str:
        """Returns label format type."""
