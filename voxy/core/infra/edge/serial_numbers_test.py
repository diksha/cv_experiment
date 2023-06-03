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

from parameterized import parameterized

from core.infra.edge.serial_numbers import (
    generate_serial_number,
    validate_serial_number,
)


class TestSerialNumberGeneration(unittest.TestCase):
    @parameterized.expand(
        [
            ({"mmm": 1, "yy": 21, "ww": 11, "nnn": 1}, "00121110016"),
            ({"mmm": 2, "yy": 20, "ww": 5, "nnn": 99}, "00220050998"),
            ({"mmm": 30, "yy": 2, "ww": 25, "nnn": 399}, "03002253991"),
            ({"mmm": 400, "yy": 8, "ww": 3, "nnn": 399}, "40008033995"),
        ]
    )
    def test_generate_serial_number(self, input_kwargs, expected_output):
        self.assertEqual(
            generate_serial_number(**input_kwargs), expected_output
        )

    @parameterized.expand(
        [
            ("00121110016", True),
            ("00220050998", True),
            ("03002253991", True),
            ("40008033995", True),
            ("40008033994", False),
            ("03002353991", False),
            ("3002253991", False),
            ("32253991", False),
        ]
    )
    def test_validate_serial_number(self, serial_number, expected_valid):
        if expected_valid:
            msg = f"Serial number should be valid: {serial_number}"
        else:
            msg = f"Serial number should NOT be valid: {serial_number}"
        self.assertEqual(
            validate_serial_number(serial_number), expected_valid, msg
        )

    def test_serial_numbers_must_be_11_digits(self):
        self.assertEqual(validate_serial_number("00121110016"), True)
        self.assertEqual(validate_serial_number("0300235399"), False)
        self.assertEqual(validate_serial_number("030023539911"), False)


if __name__ == "__main__":
    unittest.main()
