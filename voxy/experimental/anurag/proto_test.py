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
from core.structs.states import BaseState
from core.structs.protobufs.states_pb2 import BaseState as BaseStatePb

state = BaseState(timestamp_ms=1, camera_uuid="a", actor_id="1")
proto = state.to_proto()
f = open("/tmp/test.pb", "wb")
f.write(proto.SerializeToString())
f.close()

f1 = open("/tmp/test.pb", "rb")
proto2 = BaseStatePb()
proto2.ParseFromString(f1.read())
print(proto2)
