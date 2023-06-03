from datetime import datetime, timedelta
from typing import final

from protos.platform.bowser.v1.bowser_config_keys_pb2 import Unit, seconds
from services.platform.bowser.engine.utils.proto_utils import (
    ProtoException,
    ProtoUtils,
)

# trunk-ignore-all(pylint/E0611): ignore pb import errors


class KeyUtils:
    @staticmethod
    @final
    def get_timestamp_granularity(epoch_milli_timestamp: float, modulo: int):
        """Allow to get on 2 different key the Date and the Time of an epoch timestamp

        :param float epoch_milli_timestamp: epoch mili timestamp
        :param int modulo: Represent number of second to modulo
        :raises ProtoException: when module is higher than a Day
        :returns: Date and Time of the day
        :rtype: Tuple

        """

        if modulo > ProtoUtils.get_extension_by_enum(
            Unit, seconds, Unit.UNIT_DAY
        ):
            raise ProtoException(
                "Impossible to manipulate a epoch time with a module bigger than a Day"
            )

        # Transform to Seconds
        timestamp_s = epoch_milli_timestamp / 1000
        # Apply Modulo
        timestamp_s = timestamp_s - (timestamp_s % modulo)
        # dissociate timestamp Date and Time
        # Create two fields rather than 1 DateTime ( Better modularity )

        return [
            datetime.fromtimestamp(timestamp_s).strftime("%Y-%m-%d"),
            datetime.fromtimestamp(timestamp_s).strftime("%H:%M:%S"),
        ]

    @staticmethod
    @final
    def get_timestamp_granularity_by_proto_unit(
        epoch_milli_timestamp: float, unit: Unit
    ):
        """Allow to get on 2 different key the Date and the Time of an epoch timestamp

        :param float epoch_milli_timestamp: epoch milli timestamp
        :param Unit unit: Represent number of second to modulo by proto
        :returns: Date and Time of the day
        :rtype: Tuple
        """

        modulo = ProtoUtils.get_extension_by_enum(Unit, seconds, unit)

        if unit == Unit.UNIT_WEEK:
            date_str = datetime.fromtimestamp(
                epoch_milli_timestamp / 1000
            ).strftime("%Y-%m-%d")
            date_time = datetime.strptime(date_str, "%Y-%m-%d")
            start_of_week = date_time - timedelta(days=date_time.weekday())
            return [start_of_week.strftime("%Y-%m-%d")]

        if unit == Unit.UNIT_MONTH:
            return [
                datetime.fromtimestamp(epoch_milli_timestamp / 1000).strftime(
                    "%Y-%m-01"
                )
            ]

        return KeyUtils.get_timestamp_granularity(
            epoch_milli_timestamp, modulo
        )
