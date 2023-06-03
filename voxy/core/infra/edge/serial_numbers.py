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
from datetime import datetime

"""
Serial number anatomy:

    `mmmyywwnnnc`

    mmm = model (001=gaming box)
    yy = 2 digit year (i.e. 21 = 2021)
    ww = 2 digit week of year
    nnn = incrementing number per week
    c = check digit per Damm algo

Damm algorithm:
    https://en.wikipedia.org/wiki/Damm_algorithm
    https://rosettacode.org/wiki/Damm_algorithm#Python
"""

_damm_matrix = (
    (0, 3, 1, 7, 5, 9, 8, 6, 4, 2),
    (7, 0, 9, 2, 1, 5, 4, 8, 6, 3),
    (4, 2, 0, 6, 8, 7, 1, 3, 5, 9),
    (1, 7, 5, 0, 9, 8, 3, 4, 2, 6),
    (6, 1, 2, 3, 0, 4, 5, 9, 7, 8),
    (3, 6, 7, 4, 2, 0, 9, 5, 8, 1),
    (5, 8, 6, 9, 7, 2, 0, 1, 3, 4),
    (8, 9, 4, 5, 3, 6, 2, 0, 1, 7),
    (9, 4, 3, 8, 6, 1, 7, 2, 0, 5),
    (2, 5, 8, 1, 4, 3, 6, 7, 9, 0),
)


def _calculate_check_digit(serial_number: str) -> int:
    interim = 0
    for digit in serial_number:
        interim = _damm_matrix[interim][int(digit)]
    return interim


def generate_serial_number(
    *,
    mmm: int = None,
    yy: int = int(datetime.today().strftime("%y")),
    ww: int = int(datetime.today().strftime("%U")),
    nnn: int = None,
) -> str:
    if not mmm:
        raise ValueError("Model number (mmm) is required")
    elif mmm < 1 or mmm > 999:
        raise ValueError(
            f"Model number (mmm) is out of range (1-999), received: {mmm}"
        )
    if not nnn:
        raise ValueError("Incremental number (nnn) is required")
    elif nnn < 1 or nnn > 999:
        raise ValueError(
            f"Incremental number (mmm) is out of range (1-999), received: {mmm}"
        )
    mmm_pad = str(mmm).zfill(3)
    yy_pad = str(yy).zfill(2)
    ww_pad = str(ww).zfill(2)
    nnn_pad = str(nnn).zfill(3)
    serial_without_check_digit = f"{mmm_pad}{yy_pad}{ww_pad}{nnn_pad}"
    check_digit = _calculate_check_digit(serial_without_check_digit)
    return f"{serial_without_check_digit}{check_digit}"


def validate_serial_number(serial_number: str) -> bool:
    if len(serial_number) != 11:
        return False
    row = 0
    for digit in serial_number:
        row = _damm_matrix[row][int(digit)]
    return row == 0
