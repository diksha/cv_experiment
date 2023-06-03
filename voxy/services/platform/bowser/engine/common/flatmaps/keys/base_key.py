from abc import ABC, abstractmethod
from typing import Tuple

from pyflink.datastream import FlatMapFunction

from protos.platform.bowser.v1.bowser_config_keys_pb2 import ProcessorKeys

# trunk-ignore-all(pylint/E0611): ignore pb import errors


class BaseKey(FlatMapFunction, ABC):
    def __init__(self, proto_key: ProcessorKeys):
        super().__init__()
        self._key_proto = proto_key

    @abstractmethod
    def _compute_timestamp(self, raw_object):
        pass

    def _compute_key(self, raw_object, key):
        """Compute each key present on the ProcessorKeys config object

        :param Object raw_object: the generic raw object on the data stream
        :param str key: the key to take the value from

        :returns: the value to key by ( group by )
        :rtype: Object

        """
        try:
            if raw_object[key] is None:
                return f"unknow_key_{key}"
            return raw_object[key]

        except KeyError:
            return f"unknow_key_{key}"

    def get_key(self, raw_object) -> Tuple:
        """List all the key present on the ProcessorKeys config object and try to get value from it

        :param Object raw_object: the generic raw object on the data stream

        :returns: Tuple with all the keys value in it, they will become unique later
        :rtype: Tuple

        """
        value_list = []

        for key in self._key_proto.fields:
            value_list.append(self._compute_key(raw_object.to_dict(), key))

        value_list.extend(self._compute_timestamp(raw_object.to_dict()))
        return tuple(value_list)

    def flat_map(self, value) -> Tuple:
        """Transform Incident or States/Event to Keys in order to be Keyed

        :param self: Access the class attributes
        :param  VoxelObject value: Get the key from the value
        :returns: A tuple of the key and value
        """
        yield self.get_key(value)
