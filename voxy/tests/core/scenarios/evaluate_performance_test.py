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
import unittest

from core.scenarios.evaluate_performance import (
    MonitorPerformanceEvaluator,
    count_detected_incidents,
    count_ground_truth_incidents,
    evaluate_performance,
)
from core.structs.incident import Incident


class EvaluatePerformanceTest(unittest.TestCase):
    def test_count_detected_incidents(self) -> None:
        """Test different cases of count_detected_incidents"""
        # Test no incidents detected
        incidents_detected = count_detected_incidents(["PIGGYBACK"], [])
        self.assertEqual(incidents_detected["PIGGYBACK"], 0)

        # Test one incident detected
        incidents_detected = count_detected_incidents(
            ["PIGGYBACK"], [Incident(incident_type_id="PIGGYBACK")]
        )
        self.assertEqual(incidents_detected["PIGGYBACK"], 1)

        # Test cooldown incident
        incidents_detected = count_detected_incidents(
            ["PIGGYBACK"],
            [
                Incident(incident_type_id="PIGGYBACK"),
                Incident(incident_type_id="PIGGYBACK", cooldown_tag=True),
            ],
        )
        self.assertEqual(incidents_detected["PIGGYBACK"], 1)

        # Test that strings are not accepted
        with self.assertRaises(TypeError):
            count_detected_incidents(["PIGGYBACK"], ["PIGGYBACK"])

    def test_count_ground_truth_incidents(self) -> None:
        """Test count_ground_truth_incidents"""

        # Test no ground truth incidents (negatives)
        ground_truth_incident_count = count_ground_truth_incidents(
            ["PIGGYBACK"], []
        )
        self.assertEqual(ground_truth_incident_count["PIGGYBACK"], 0)

        # Test one ground truth incident
        ground_truth_incident_count = count_ground_truth_incidents(
            ["PIGGYBACK"], ["PIGGYBACK"]
        )
        self.assertEqual(ground_truth_incident_count["PIGGYBACK"], 1)

        # Test multiple ground truth of different types
        ground_truth_incident_count = count_ground_truth_incidents(
            ["PIGGYBACK", "BAD_POSTURE"],
            [
                "PIGGYBACK",
                "PIGGYBACK",
                "BAD_POSTURE",
            ],
        )
        self.assertEqual(ground_truth_incident_count["PIGGYBACK"], 2)
        self.assertEqual(ground_truth_incident_count["BAD_POSTURE"], 1)

        # Test multiple ground truth of different types, one isn't tracked
        ground_truth_incident_count = count_ground_truth_incidents(
            ["PIGGYBACK"],
            [
                "PIGGYBACK",
                "PIGGYBACK",
                "BAD_POSTURE",
            ],
        )
        self.assertEqual(ground_truth_incident_count["PIGGYBACK"], 2)
        self.assertEqual(ground_truth_incident_count["BAD_POSTURE"], 1)

    def test_monitor_performance_evaluator(self) -> None:
        """Test all public member functions of MonitorPerformanceEvaluator"""

        monitor_performance_evaluator = MonitorPerformanceEvaluator()

        # Test base case for all
        self.assertIsNone(monitor_performance_evaluator.get_fn_rate())
        self.assertIsNone(monitor_performance_evaluator.get_tp_rate())
        self.assertIsNone(monitor_performance_evaluator.get_fp_rate())
        self.assertIsNone(monitor_performance_evaluator.get_precision())
        self.assertIsNone(monitor_performance_evaluator.get_recall())

        # Add a few incidents and test all
        monitor_performance_evaluator.add_true_positive("sample_true_positive")

        monitor_performance_evaluator.add_false_positive(
            "sample_false_positive"
        )
        monitor_performance_evaluator.add_false_positive(
            "sample_false_positive"
        )

        monitor_performance_evaluator.add_true_negative("sample_true_negative")
        monitor_performance_evaluator.add_true_negative("sample_true_negative")
        monitor_performance_evaluator.add_true_negative("sample_true_negative")

        monitor_performance_evaluator.add_false_negative(
            "sample_false_negative"
        )
        monitor_performance_evaluator.add_false_negative(
            "sample_false_negative"
        )
        monitor_performance_evaluator.add_false_negative(
            "sample_false_negative"
        )
        monitor_performance_evaluator.add_false_negative(
            "sample_false_negative"
        )

        self.assertEqual(
            monitor_performance_evaluator.get_fn_rate(), 4.0 / (4 + 1)
        )
        self.assertEqual(
            monitor_performance_evaluator.get_tp_rate(), 1.0 / (1 + 4)
        )
        self.assertEqual(
            monitor_performance_evaluator.get_fp_rate(), 2.0 / (2 + 3)
        )
        self.assertEqual(
            monitor_performance_evaluator.get_precision(), 1.0 / (1 + 2)
        )
        self.assertEqual(
            monitor_performance_evaluator.get_recall(), 1.0 / (1 + 4)
        )

    def test_evaluate_performance(self) -> None:
        """Test all branches of evaluate_performance"""

        # Test when num scenarios and detected incidents are not equal
        performance_by_incident = evaluate_performance(
            [], [[Incident(incident_type_id="PIGGYBACK")]]
        )
        self.assertFalse(performance_by_incident)

        # test one monitor, 1 correct positive
        mocked_scenario_1 = {
            "config": {
                "incident": {"state_machine_monitors_requested": ["PIGGYBACK"]}
            },
            "incidents": ["PIGGYBACK"],
            "scenario_for_incidents": ["PIGGYBACK"],
            "video_uuid": "sample_video_uuid",
        }
        performance_by_incident = evaluate_performance(
            [mocked_scenario_1], [[Incident(incident_type_id="PIGGYBACK")]]
        )
        self.assertEqual(performance_by_incident["PIGGYBACK"].get_fn_rate(), 0)
        self.assertEqual(performance_by_incident["PIGGYBACK"].get_tp_rate(), 1)
        self.assertIsNone(performance_by_incident["PIGGYBACK"].get_fp_rate())
        self.assertEqual(
            performance_by_incident["PIGGYBACK"].get_precision(), 1
        )
        self.assertEqual(performance_by_incident["PIGGYBACK"].get_recall(), 1)

        # test false positive
        performance_by_incident = evaluate_performance(
            [mocked_scenario_1],
            [
                [
                    Incident(incident_type_id="PIGGYBACK"),
                    Incident(incident_type_id="PIGGYBACK"),
                ]
            ],
        )
        self.assertEqual(performance_by_incident["PIGGYBACK"].get_fn_rate(), 0)
        self.assertEqual(performance_by_incident["PIGGYBACK"].get_tp_rate(), 1)
        self.assertEqual(performance_by_incident["PIGGYBACK"].get_fp_rate(), 1)
        self.assertEqual(
            performance_by_incident["PIGGYBACK"].get_precision(), 0.5
        )
        self.assertEqual(performance_by_incident["PIGGYBACK"].get_recall(), 1)

        # test false negative
        performance_by_incident = evaluate_performance(
            [mocked_scenario_1], [[]]
        )
        self.assertEqual(performance_by_incident["PIGGYBACK"].get_fn_rate(), 1)
        self.assertEqual(performance_by_incident["PIGGYBACK"].get_tp_rate(), 0)
        self.assertIsNone(
            performance_by_incident["PIGGYBACK"].get_fp_rate(), 0
        )
        self.assertIsNone(performance_by_incident["PIGGYBACK"].get_precision())
        self.assertEqual(performance_by_incident["PIGGYBACK"].get_recall(), 0)

        # test true negative
        mocked_scenario_2 = {
            "config": {
                "incident": {
                    "state_machine_monitors_requested": [
                        "piggyback",
                    ]
                }
            },
            "incidents": [],
            "scenario_for_incidents": ["PIGGYBACK"],
            "video_uuid": "video_uuid",
        }
        performance_by_incident = evaluate_performance(
            [mocked_scenario_2], [[Incident(incident_type_id="PIGGYBACK")]]
        )
        self.assertIsNone(performance_by_incident["PIGGYBACK"].get_fn_rate())
        self.assertIsNone(performance_by_incident["PIGGYBACK"].get_tp_rate())
        self.assertEqual(performance_by_incident["PIGGYBACK"].get_fp_rate(), 1)
        self.assertEqual(
            performance_by_incident["PIGGYBACK"].get_precision(), 0
        )
        self.assertIsNone(performance_by_incident["PIGGYBACK"].get_recall())


if __name__ == "__main__":
    unittest.main()
