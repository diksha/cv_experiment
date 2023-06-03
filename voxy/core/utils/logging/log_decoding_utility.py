from core.structs.frame import Frame
from core.structs.vignette import Vignette


class LogDecodingUtility:
    @classmethod
    def decode_as_timestamped_dictionary(cls, x):
        return lambda items: {
            cls.get_timestamp(item): x(item) for item in items
        }

    @classmethod
    def get_log_decode_map(cls):
        return {
            "TemporalNode": cls.decode_as_timestamped_dictionary(
                Frame.from_dict
            ),
            "AcausalNode": cls.decode_as_timestamped_dictionary(
                Vignette.from_dict
            ),
            "StateNode": cls.decode_as_timestamped_dictionary(
                Vignette.from_dict
            ),
        }

    @classmethod
    def get_timestamp(cls, item):
        epoch_timestamp_ms = item.get("epoch_timestamp_ms")
        current_frame_struct = item.get("present_frame_struct")
        if epoch_timestamp_ms is not None:
            return epoch_timestamp_ms
        if current_frame_struct is not None:
            present_timestamp_ms = current_frame_struct.get(
                "epoch_timestamp_ms"
            )
            return present_timestamp_ms
        return 0

    @classmethod
    def decode(cls, node_logs):
        decoded_logs = {}
        LOG_DECODE_MAP = cls.get_log_decode_map()
        for node_name in node_logs:
            decoded_logs[node_name] = LOG_DECODE_MAP[node_name](
                node_logs[node_name]
            )
        return decoded_logs
