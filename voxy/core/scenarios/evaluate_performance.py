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
import logging
from collections import Counter
from typing import Any, Dict, List, Union

from core.structs.incident import Incident

# TODO(Vai): Remove this when task PERCEPTION-92 is completed.
monitor_incident_map = {
    "piggyback": "PIGGYBACK",
    "door_violation": "DOOR_VIOLATION",
    "door_intersection": "NO_STOP_AT_DOOR_INTERSECTION",
    "bad_posture": "BAD_POSTURE",
    "hard_hat": "HARD_HAT",
    "safety_vest": "SAFETY_VEST",
    "parking": "PARKING_DURATION",
    "open_door": "OPEN_DOOR_DURATION",
    "no_stop_at_aisle_end": "NO_STOP_AT_END_OF_AISLE",
    "no_ped_zone": "NO_PED_ZONE",
    "intersection": "NO_STOP_AT_INTERSECTION",
    "overreaching": "OVERREACHING",
    "spill": "SPILL",
    "production_line_down": "PRODUCTION_LINE_DOWN",
    "high_vis_hat_or_vest": "HIGH_VIS_HAT_OR_VEST",
    "bump_cap": "BUMP_CAP",
    "bad_posture_with_uniform": "BAD_POSTURE_WITH_SAFETY_UNIFORM",
    "overreaching_with_uniform": "OVERREACHING_WITH_SAFETY_UNIFORM",
    "n_person_ped_zone": "N_PERSON_PED_ZONE",
    "obstruction": "OBSTRUCTION",
}


class MonitorPerformanceEvaluator:
    def __init__(self):

        self.true_positives = []
        self.false_positives = []
        self.true_negatives = []
        self.false_negatives = []

    # Adders
    def add_true_positive(self, scenario):
        self.true_positives.append(scenario)

    def add_false_positive(self, scenario):
        self.false_positives.append(scenario)

    def add_true_negative(self, scenario):
        self.true_negatives.append(scenario)

    def add_false_negative(self, scenario):
        self.false_negatives.append(scenario)

    # Getters

    def get_fn_rate(self):

        num_positive_gt = len(self.true_positives) + len(self.false_negatives)

        if num_positive_gt > 0:
            fn_rate = len(self.false_negatives) / num_positive_gt
            return fn_rate

        return None

    def get_tp_rate(self):

        num_positive_gt = len(self.true_positives) + len(self.false_negatives)

        if num_positive_gt > 0:
            tp_rate = len(self.true_positives) / num_positive_gt
            return tp_rate

        return None

    def get_fp_rate(self):

        num_positive_gt = len(self.false_positives) + len(self.true_negatives)

        if num_positive_gt > 0:
            fp_rate = len(self.false_positives) / num_positive_gt
            return fp_rate

        return None

    def get_precision(self):

        num_positive_detected = len(self.true_positives) + len(
            self.false_positives
        )

        if num_positive_detected > 0:
            precision = len(self.true_positives) / num_positive_detected
            return precision

        return None

    def get_recall(self):

        num_positive_gt = len(self.true_positives) + len(self.false_negatives)

        if num_positive_gt > 0:
            recall = len(self.true_positives) / num_positive_gt
            return recall

        return None


def count_ground_truth_incidents(incidents_requested, ground_truth) -> dict:
    """Given incidents requested and ground truth, count the number of ground truths

    Args:
        incidents_requested (List): requested incidents
        ground_truth (List): ground truth from scenario

    Returns:
        dict: incident to ground truth count
    """
    ground_truth_incident_count = {}

    if ground_truth:
        for incident_type in ground_truth:
            if incident_type not in ground_truth_incident_count:
                ground_truth_incident_count[incident_type] = 1
            else:
                ground_truth_incident_count[incident_type] += 1
    else:
        for incident in incidents_requested:
            ground_truth_incident_count[incident] = 0

    return ground_truth_incident_count


def count_detected_incidents(
    incidents_requested: List[str],
    detected_incidents: List[Incident],
) -> Dict[str, int]:
    """Count incidents by incident type id

    Args:
        incidents_requested: A list of incident type ids requested
        detected_incidents: A list of detected incidents of type Incident.

    Returns:
        A dictionary mapping incident type id to number of such incidents
        identified
    Raises:
        TypeError: if the detected incident is not of type Incident
    """
    detected_incident_count = Counter()

    # This takes care of the case where there are multiple requested incidents
    # but incidents aren't generated for all of them. Since we're using Counter,
    # we don't technically need to initialize the counts to 0, but this will
    # ensure calling .keys() on the return value gives all incident types.
    for incident in incidents_requested:
        detected_incident_count[incident] = 0

    if detected_incidents:
        for incident in detected_incidents:
            if not isinstance(incident, Incident):
                raise TypeError("incidents should all be of type Incident.")

            incident_type_id = incident.incident_type_id
            if (
                incident_type_id in detected_incident_count
                and incident.cooldown_tag is False
            ):
                detected_incident_count[incident_type_id] += 1

    return detected_incident_count


# trunk-ignore-begin(pylint/R0914)
def evaluate_performance(
    scenarios: List[Dict[str, Any]],
    detected_incidents: List[Union[str, Incident]],
) -> Dict[str, MonitorPerformanceEvaluator]:
    """Summarize the performance of perception based on ground truth and inferences

    Also, optionally, store the results of the performance evaluation in Postgres
    for future reference.

    Args:
        scenarios: the scenarios inference was run on, including ground truth
        detected_incidents: The list of incidents that perception found in the scenarios.
            The list elements can be either an Incident struct or just the string for the
            incident type id.

    Returns:
        The performance results, as a dict from incident type id to
        MonitorPerformanceEvaluator.
    """

    performance_by_incident = {}

    # Evaluate Sensitivity & Specificity
    if len(scenarios) != len(detected_incidents):
        logging.info("Error! Ground truth and result length are not equal.")
        return performance_by_incident

    for i, scenario in enumerate(scenarios):

        ground_truth = scenario["incidents"]

        incidents_requested = scenario["scenario_for_incidents"]

        # Count each type of ground truth incident
        ground_truth_incident_count = count_ground_truth_incidents(
            incidents_requested, ground_truth
        )

        # Count each type of detected incident
        detected_incident_count = count_detected_incidents(
            incidents_requested, detected_incidents[i]
        )

        # add to monitor list
        for monitor, ground_truth_count in ground_truth_incident_count.items():

            # Create performance tracker if it does not exist
            if monitor not in performance_by_incident:
                performance_by_incident[
                    monitor
                ] = MonitorPerformanceEvaluator()

            if ground_truth_count != detected_incident_count[monitor]:
                logging.info(
                    "Failed for %s %s detected %d ground truth %d",
                    scenario["video_uuid"],
                    monitor,
                    detected_incident_count[monitor],
                    ground_truth_count,
                )
            # True Negative
            if (
                detected_incident_count[monitor] == 0
                and ground_truth_count == 0
            ):
                performance_by_incident[monitor].add_true_negative(scenario)

            elif ground_truth_count >= detected_incident_count[monitor]:
                # True Positive
                for _ in range(detected_incident_count[monitor]):
                    performance_by_incident[monitor].add_true_positive(
                        scenario
                    )
                # False Negative
                for _ in range(
                    ground_truth_count - detected_incident_count[monitor]
                ):
                    performance_by_incident[monitor].add_false_negative(
                        scenario
                    )

            # False Positive
            elif ground_truth_count < detected_incident_count[monitor]:
                for _ in range(
                    detected_incident_count[monitor] - ground_truth_count
                ):
                    performance_by_incident[monitor].add_false_positive(
                        scenario
                    )
                # True Positive
                for _ in range(ground_truth_count):
                    performance_by_incident[monitor].add_true_positive(
                        scenario
                    )

    # return output
    return performance_by_incident
    # trunk-ignore-end(pylint/R0914)


def log_performance_results(performance_by_incident):
    """Logs performance results to the screen

    Args:
        performance_by_incident (dict): dictionary of incident types and the performance
    """
    # Print results
    for key, value in performance_by_incident.items():
        fn_rate = value.get_fn_rate()
        tp_rate = value.get_tp_rate()
        fp_rate = value.get_fp_rate()
        recall = value.get_recall()
        precision = value.get_precision()

        # Track only Video UUID
        false_positive_uuids = [
            item["video_uuid"] for item in value.false_positives
        ]
        false_negative_uuids = [
            item["video_uuid"] for item in value.false_negatives
        ]

        logging.info(
            f"False Positives: {false_positive_uuids} \
            \n False Negatives: {false_negative_uuids}"
        )

        logging.info(
            f"Number of False Positives: {len(value.false_positives)} \
            \n Number of False Negatives: {len(value.false_negatives)} \
            \n Number of True Positives: {len(value.true_positives)} \
            \n Number of True Negatives: {len(value.true_negatives)}"
        )

        logging.info(
            f"Monitor: {key} \
            \n Precision: {precision} \
            \n Recall: {recall} \
            \n True Positive Rate:  {tp_rate}\
            \n False Positive Rate: {fp_rate}\
            \n False Negative Rate: {fn_rate} "
        )


def compare_performance(
    performance_by_incident, compare_performance_by_incident
):
    incident_intersection = set(performance_by_incident.keys()).intersection(
        set(compare_performance_by_incident.keys())
    )

    for incident_key in incident_intersection:
        incident_performance = performance_by_incident[incident_key]
        compare_incident_performance = compare_performance_by_incident[
            incident_key
        ]

        # Track only Video UUID
        false_positive_uuids = {
            item["video_uuid"] for item in incident_performance.false_positives
        }
        false_negative_uuids = {
            item["video_uuid"] for item in incident_performance.false_negatives
        }

        compare_false_positive_uuids = {
            item["video_uuid"]
            for item in compare_incident_performance.false_positives
        }

        compare_false_negative_uuids = {
            item["video_uuid"]
            for item in compare_incident_performance.false_negatives
        }

        # Calculate diffs
        unique_false_positives = false_positive_uuids.difference(
            compare_false_positive_uuids
        )
        unique_false_negatives = false_negative_uuids.difference(
            compare_false_negative_uuids
        )

        compare_unique_false_positives = (
            compare_false_positive_uuids.difference(false_positive_uuids)
        )
        compare_unique_false_negatives = (
            compare_false_negative_uuids.difference(false_negative_uuids)
        )

        logging.info(
            f"Unique False Positives: {unique_false_positives} \
            \n Compare False Positives: {compare_unique_false_positives} \
            \n Unique False Negatives: {unique_false_negatives} \
            \n Compare False Negatives: {compare_unique_false_negatives}"
        )


if __name__ == "__main__":
    print("Not an executable, just a library.")
