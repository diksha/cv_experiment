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


class BaseStateGenerator:
    def process_vignette(self, vignette):
        raise NotImplementedError(
            "Generators must implement process_vignette."
        )

    def _get_end_timestamp_ms(self, vignette):
        if len(vignette.future_frame_structs):
            return vignette.future_frame_structs[0].relative_timestamp_ms
        return vignette.present_timestamp_ms
