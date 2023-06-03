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

from core.perception.acausal.algorithms.association_controller import (
    AssociationAlgorithmController,
)
from core.perception.acausal.algorithms.discrete_bayes import (
    DiscreteBayesAlgorithm,
)
from core.perception.acausal.algorithms.ergonomics_smoothing import (
    ErgonomicsSmoothingAlgorithm,
)
from core.perception.acausal.algorithms.is_stationary import (
    IsStationaryAlgorithm,
)
from core.perception.acausal.algorithms.motion_zone_smoothing import (
    MotionZoneSmoothing,
)
from core.perception.acausal.algorithms.ppe_smoothening import (
    PPESmootheningAlgorithm,
)
from core.perception.acausal.algorithms.proximity_controller import (
    ProximityAlgorithmController,
)
from core.perception.acausal.algorithms.velocity_estimation import (
    VelocityEstimationAlgorithm,
)


class AcausalController:
    _INCIDENT_TYPE_ALGORITHM_DEPENDENCY_GRAPH = {
        "parking": [VelocityEstimationAlgorithm, IsStationaryAlgorithm],
        "hard_hat": [PPESmootheningAlgorithm],
        "safety_vest": [PPESmootheningAlgorithm],
        "door_intersection": [VelocityEstimationAlgorithm],
        "intersection": [VelocityEstimationAlgorithm],
        "no_stop_at_aisle_end": [VelocityEstimationAlgorithm],
        "bad_posture": [ErgonomicsSmoothingAlgorithm],
        "overreaching": [ErgonomicsSmoothingAlgorithm],
        "ergo2_bad_posture": [ErgonomicsSmoothingAlgorithm],
        "door_violation": [DiscreteBayesAlgorithm],
        "open_door": [DiscreteBayesAlgorithm],
        "piggyback": [DiscreteBayesAlgorithm],
        "production_line_down": [MotionZoneSmoothing],
        "high_vis_hat_or_vest": [PPESmootheningAlgorithm],
        "bump_cap": [PPESmootheningAlgorithm],
        "bad_posture_with_uniform": [
            ErgonomicsSmoothingAlgorithm,
            PPESmootheningAlgorithm,
        ],
        "overreaching_with_uniform": [
            ErgonomicsSmoothingAlgorithm,
            PPESmootheningAlgorithm,
        ],
        "n_person_ped_zone": [VelocityEstimationAlgorithm],
        "pit_near_miss": [ProximityAlgorithmController],
        "obstruction": [VelocityEstimationAlgorithm, IsStationaryAlgorithm],
    }

    _ORDERED_ALGORITHMS = [
        VelocityEstimationAlgorithm,
        IsStationaryAlgorithm,
        PPESmootheningAlgorithm,
        DiscreteBayesAlgorithm,
        AssociationAlgorithmController,
        ProximityAlgorithmController,
        ErgonomicsSmoothingAlgorithm,
        MotionZoneSmoothing,
    ]

    def __init__(self, config):
        self._config = config
        self._ordered_algorithms = self.get_ordered_algorithms(config)

    def get_ordered_algorithms(self, config: dict) -> list:
        # Initialize based on config.
        # Only run algorithms requested in config.
        # create a new list to prevent modifying the original config by reference.
        all_incident_types_requested = list(
            config.get("incident", {}).get(
                "state_machine_monitors_requested", []
            )
        )
        all_incident_types_requested.extend(
            config.get("incident", {}).get("monitors_requested", [])
        )
        all_incident_types_requested.extend(
            config.get("incident", {}).get("compliance_metrics", [])
        )
        # TODO(harishma): Replace hardcoded string names with names from some global config.
        # WARNING: These checks can lead to unintended behavior if any downstream incidents start using
        # any states that they currently do not use. These checks should ideally be accompanied by explicit
        # publisher subscriber contracts. That is, a downstream user of an acausal algorithms should subscribe
        # to it explicitly and validate it at runtime.
        required_algorithms = {
            AssociationAlgorithmController,
        }
        for incident_type in all_incident_types_requested:
            for (
                required_algorithm
            ) in AcausalController._INCIDENT_TYPE_ALGORITHM_DEPENDENCY_GRAPH.get(
                incident_type, []
            ):
                required_algorithms.add(required_algorithm)

        ordered_algorithms = [
            algo(self._config)
            for algo in AcausalController._ORDERED_ALGORITHMS
            if algo in required_algorithms
        ]

        return ordered_algorithms

    def process_vignette(self, vignette):
        return self._process(vignette)

    def _process(self, vignette):
        for algorithm in self._ordered_algorithms:
            vignette = algorithm.process_vignette(vignette)
        return vignette

    def finalize(self):
        pass
