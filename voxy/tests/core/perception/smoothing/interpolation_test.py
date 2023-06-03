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

from core.perception.smoothing.interpolation import sample_interpolated_signal


class InterpolationTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_sampling(self) -> None:
        # guassian smoothing
        test_timestamps = [0, 1000]
        signal = [1000, 0]
        sampled_signal, _ = sample_interpolated_signal(
            signal, test_timestamps, 500
        )
        # if we interpolate we expect this to have three values
        self.assertEqual(1000, sampled_signal[0])
        self.assertEqual(500, sampled_signal[1])
        self.assertEqual(0, sampled_signal[2])


if __name__ == "__main__":
    unittest.main()
