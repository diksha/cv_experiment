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
class TemporalViolationTracker:
    MAX_AGE = 20
    MAX_CREDIT = 30

    def __init__(self, minimum_filter_threshold=4):
        self.minimum_filter_threshold = minimum_filter_threshold
        # smooth out the count. alpha is in [0, 1]. larger alpha means less smoothing
        self.penalty = {}
        self.ages = {}
        self.violators_filtered = []

    def get_violators(self, violators_track_id, non_violators_track_id={}):
        # Add credit for non-violators
        for track_id in non_violators_track_id:
            self.penalty[track_id] = (
                max(self.penalty[track_id] - 1, -1 * self.MAX_CREDIT)
                if track_id in self.penalty
                else -1
            )
            self.ages[track_id] = 0

        # Add penalty for violators
        for track_id in violators_track_id:
            self.penalty[track_id] = (
                self.penalty[track_id] + 1 if track_id in self.penalty else 1
            )
            self.ages[track_id] = 0

        for track_id in self.penalty:
            self.ages[track_id] += 1

        self.violators_filtered = [
            track_id
            for track_id in self.penalty
            if self.penalty[track_id] > self.minimum_filter_threshold
        ]

        # garbage collection: remove old ones
        small_keys = [
            track_id
            for track_id in self.penalty
            if self.ages[track_id] > self.MAX_AGE
        ]
        [self.penalty.pop(key) for key in small_keys]
        [self.ages.pop(key) for key in small_keys]

        return self.violators_filtered, self.penalty

    def reset_violators(self):
        [self.penalty.pop(key) for key in self.violators_filtered]
        [self.ages.pop(key) for key in self.ages]

    def reset_single_violator(self, track_id):
        self.penalty.pop(track_id)
        self.ages.pop(track_id)
