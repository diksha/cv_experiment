from abc import ABC
from typing import Tuple

from services.platform.bowser.engine.common.flatmaps.keys.base_key import (
    BaseKey,
)


class StateEventRichKey(BaseKey, ABC):
    def get_key(self, raw_object) -> Tuple:
        raise NotImplementedError(
            "This State Event Keys is not yet implemented"
        )
