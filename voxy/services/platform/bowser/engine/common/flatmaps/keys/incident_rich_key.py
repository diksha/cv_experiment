from abc import ABC

from services.platform.bowser.engine.common.flatmaps.keys.base_key import (
    BaseKey,
)
from services.platform.bowser.engine.utils.key_utils import KeyUtils


class IncidentRichKey(BaseKey, ABC):
    def _compute_timestamp(self, raw_object):
        if self._key_proto.timestamp.field is not None:
            value = KeyUtils.get_timestamp_granularity_by_proto_unit(
                raw_object[self._key_proto.timestamp.field],
                self._key_proto.timestamp.by,
            )
        return value
