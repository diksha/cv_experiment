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
import random
import unittest

from sematic.resolvers.silent_resolver import SilentResolver

from core.infra.sematic.perception.performance_evaluation import (
    EvaluationResult,
    EvaluationResults,
    evaluate_performance,
)
from core.structs.incident import Incident
from core.structs.scenario import Scenario

EXAMPLE_CAMERA_UUID = "americold/modesto/0001/cha"


# trunk-ignore-begin(pylint/C0115,pylint/C0116,pylint/W9011,pylint/W9012)
class PerformanceEvaluationTest(unittest.TestCase):
    def test_evaluate_performance(self):
        perfect_results = make_inferences(
            n_true_positives=100,
            n_true_negatives=100,
            n_false_positives=0,
            n_false_negatives=0,
        )
        result: EvaluationResults = evaluate_performance(
            perfect_results, "fake_run_uuid", "fake_cache_key", None
        ).resolve(SilentResolver())

        self.assertIsInstance(result, EvaluationResults)

        self.assertIsInstance(
            result.results_by_incident_type["PIGGYBACK"], EvaluationResult
        )
        self.assertEqual(
            1.0, result.results_by_incident_type["PIGGYBACK"].precision
        )
        self.assertEqual(
            1.0, result.results_by_incident_type["PIGGYBACK"].recall
        )
        self.assertEqual(
            0.0,
            result.results_by_incident_type["PIGGYBACK"].false_positive_rate,
        )
        self.assertEqual(
            1.0,
            result.results_by_incident_type["PIGGYBACK"].true_positive_rate,
        )
        self.assertEqual(
            0.0,
            result.results_by_incident_type["PIGGYBACK"].false_negative_rate,
        )

    def test_evaluate_performance_always_positive(self):
        always_positive = make_inferences(
            n_true_positives=100,
            n_true_negatives=0,
            n_false_positives=100,
            n_false_negatives=0,
        )
        result: EvaluationResults = evaluate_performance(
            always_positive, "fake_run_uuid", "fake_cache_key", None
        ).resolve(SilentResolver())
        self.assertIsInstance(result, EvaluationResults)
        self.assertIsInstance(
            result.results_by_incident_type["PIGGYBACK"], EvaluationResult
        )
        self.assertEqual(
            0.5, result.results_by_incident_type["PIGGYBACK"].precision
        )
        self.assertEqual(
            1.0, result.results_by_incident_type["PIGGYBACK"].recall
        )
        self.assertEqual(
            1.0,
            result.results_by_incident_type["PIGGYBACK"].false_positive_rate,
        )
        self.assertEqual(
            1.0,
            result.results_by_incident_type["PIGGYBACK"].true_positive_rate,
        )
        self.assertEqual(
            0.0,
            result.results_by_incident_type["PIGGYBACK"].false_negative_rate,
        )

    def test_evaluate_performance_coinflip(self):
        # assume incidents are in roughly half of scenarios,
        # this emulates perception that determines whether there
        # is an incident or not by "flipping a coin"
        coin_flip_inference = make_inferences(
            n_true_positives=100,
            n_true_negatives=100,
            n_false_positives=100,
            n_false_negatives=100,
        )
        result: EvaluationResults = evaluate_performance(
            coin_flip_inference, "fake_run_uuid", "fake_cache_key", None
        ).resolve(SilentResolver())
        self.assertIsInstance(result, EvaluationResults)
        self.assertIsInstance(
            result.results_by_incident_type["PIGGYBACK"], EvaluationResult
        )
        self.assertEqual(
            0.5, result.results_by_incident_type["PIGGYBACK"].precision
        )
        self.assertEqual(
            0.5, result.results_by_incident_type["PIGGYBACK"].recall
        )


def make_inferences(
    n_true_positives, n_true_negatives, n_false_positives, n_false_negatives
):
    fake_uuid_index = 0

    # trunk-ignore-begin(pylint/C0103)
    def make_n_inferences(n, start_index, incidents, inferred_incidents):
        return [
            Scenario(
                camera_uuid=EXAMPLE_CAMERA_UUID,
                incidents=incidents,
                video_uuid=str(i),
                inferred_incidents=inferred_incidents,
                scenario_for_incidents=["PIGGYBACK"],
            )
            for i in range(start_index, start_index + n)
        ], start_index + n
        # trunk-ignore-end(pylint/C0103)

    true_positives, fake_uuid_index = make_n_inferences(
        n=n_true_positives,
        start_index=fake_uuid_index,
        incidents=["PIGGYBACK"],
        inferred_incidents=[Incident(incident_type_id="PIGGYBACK")],
    )

    true_negatives, fake_uuid_index = make_n_inferences(
        n=n_true_negatives,
        start_index=fake_uuid_index,
        incidents=[],
        inferred_incidents=[],
    )

    false_positives, fake_uuid_index = make_n_inferences(
        n=n_false_positives,
        start_index=fake_uuid_index,
        incidents=[],
        inferred_incidents=[Incident(incident_type_id="PIGGYBACK")],
    )

    false_negatives, fake_uuid_index = make_n_inferences(
        n=n_false_negatives,
        start_index=fake_uuid_index,
        incidents=["PIGGYBACK"],
        inferred_incidents=[],
    )

    inferences = (
        true_positives + true_negatives + false_positives + false_negatives
    )
    random.shuffle(inferences)
    return inferences


# trunk-ignore-end(pylint/C0115,pylint/C0116,pylint/W9011,pylint/W9012)
