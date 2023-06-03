import base64
import io
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from botocore.response import StreamingBody
from google.protobuf.any_pb2 import Any

from core.portal.perceived_data.enums import (
    PerceivedActorStateDurationCategory,
)
from core.portal.perceived_data.perceived_actor_state_duration_batch_processor import (
    PerceivedActorStateDurationBatchProcessor,
)
from core.structs.protobufs.v1.event_pb2 import Event as PerceivedActorEvent

# trunk-ignore-all(pylint/E0611)
from core.structs.protobufs.v1.state_pb2 import State as PerceivedActorState
from protos.perception.structs.v1.actor_pb2 import ActorCategory


# trunk-ignore-all(pylint/W0212)
# trunk-ignore-all(pylint/C0116)
class TestPerceivedActorStateDurationBatchProcessor(unittest.TestCase):
    def test_execute(self):
        # TODO: Write a comprehensive test
        obj = PerceivedActorStateDurationBatchProcessor()
        obj._load_batch = MagicMock()
        obj._process_batch = MagicMock()
        obj._publish_time_buckets_as_aggregates = MagicMock()

        mock_bucket_name = "mock_bucket"
        mock_batch_key = "mock/batch/key"
        obj.execute(bucket_name=mock_bucket_name, batch_key=mock_batch_key)

    def test_calculate_time_bucket_durations(self):
        """Test for > 2 time buckets"""
        # 2/7/23 22:57:56 -> 2/8/23 01:18:40
        mock_start_time = datetime(2023, 2, 7, 22, 57, 56, 851000)
        mock_end_time = datetime(2023, 2, 8, 1, 18, 40, 392101)
        results = PerceivedActorStateDurationBatchProcessor._calculate_time_bucket_durations(
            start_time=mock_start_time, end_time=mock_end_time
        )
        expected = {
            # 2/7/23 22:00: ~4 minutes
            datetime(2023, 2, 7, 22, 0): timedelta(
                seconds=123, microseconds=149000
            ),
            # 2/7/23 23:00: 1 hour
            datetime(2023, 2, 7, 23, 0): timedelta(seconds=3600),
            # 2/8/23 00:00: 1 hour
            datetime(2023, 2, 8, 0, 0): timedelta(seconds=3600),
            # 2/8/23 1:00: ~18 minutes
            datetime(2023, 2, 8, 1, 0): timedelta(
                seconds=1120, microseconds=392101
            ),
        }
        self.assertEqual(expected, results)

    def test_process_batch(self):
        base_start_time_ms = 1675810676851
        mock_camera_uuid_1 = "mock_camera_uuid_1"
        mock_camera_uuid_2 = "mock_camera_uuid_2"
        # Message 1: (camera_1, pit stationary, 93 seconds within bucket one bucket)
        mock_any_1 = Any()
        mock_message_1 = PerceivedActorState(
            camera_uuid=mock_camera_uuid_1,
            timestamp_ms=base_start_time_ms,
            end_timestamp_ms=base_start_time_ms + 93000,
            actor_category=ActorCategory.ACTOR_CATEGORY_PIT,
            pit_is_stationary=True,
        )
        mock_any_1.Pack(mock_message_1)
        # Message 2: (camera_1, pit stationary, 7 minutes over two buckets
        # (overlap with bucket from messsage 1))
        mock_any_2 = Any()
        mock_message_2 = PerceivedActorState(
            camera_uuid=mock_camera_uuid_1,
            timestamp_ms=base_start_time_ms + 56000,
            end_timestamp_ms=base_start_time_ms + 420000,
            actor_category=ActorCategory.ACTOR_CATEGORY_PIT,
            pit_is_stationary=True,
        )
        mock_any_2.Pack(mock_message_2)
        # Message 3: (camera_1, person time (before change), ~3.5 hours over three buckets)
        mock_any_3 = Any()
        mock_message_3 = PerceivedActorState(
            camera_uuid=mock_camera_uuid_1,
            timestamp_ms=base_start_time_ms + 56000,
            end_timestamp_ms=base_start_time_ms + 13320000,
            actor_category=ActorCategory.ACTOR_CATEGORY_PERSON,
            person_is_associated=False,
            person_lift_type=None,
            person_reach_type=False,
            run_uuid="v0.93.0:d45a7ce9-f5bb-4c32-abdc-c715adcb1463",
        )
        mock_any_3.Pack(mock_message_3)
        # Message 4: (camera_2, pit stationary ~3.5 hours over 3 buckets)
        mock_any_4 = Any()
        mock_message_4 = PerceivedActorState(
            camera_uuid=mock_camera_uuid_2,
            timestamp_ms=base_start_time_ms + 56000,
            end_timestamp_ms=base_start_time_ms + 13320000,
            actor_category=ActorCategory.ACTOR_CATEGORY_PIT,
            pit_is_stationary=True,
        )
        mock_any_4.Pack(mock_message_4)
        # Message 5: (camera_1, pit stationary, invalid timestamps)
        mock_any_5 = Any()
        mock_message_5 = PerceivedActorState(
            camera_uuid=mock_camera_uuid_1,
            timestamp_ms=base_start_time_ms,
            end_timestamp_ms=base_start_time_ms - 12,
            actor_category=ActorCategory.ACTOR_CATEGORY_PIT,
            pit_is_stationary=True,
        )
        mock_any_5.Pack(mock_message_5)
        # Message 6: (camera_2, no category, ~2 hours)
        mock_any_6 = Any()
        mock_message_6 = PerceivedActorState(
            camera_uuid=mock_camera_uuid_2,
            timestamp_ms=base_start_time_ms - 900,
            end_timestamp_ms=base_start_time_ms + 7560000,
            actor_category=ActorCategory.ACTOR_CATEGORY_PERSON,
            person_is_associated=True,
        )
        mock_any_6.Pack(mock_message_6)
        # Message 7: (camera_2, pit non-stationary, ~2 hours)
        mock_any_7 = Any()
        mock_message_7 = PerceivedActorState(
            camera_uuid=mock_camera_uuid_2,
            timestamp_ms=base_start_time_ms - 900,
            end_timestamp_ms=base_start_time_ms + 7560000,
            actor_category=ActorCategory.ACTOR_CATEGORY_PIT,
            person_is_associated=True,
            pit_is_stationary=None,
        )
        mock_any_7.Pack(mock_message_7)
        # Message 8: (camera_1, event message, ~2 hours over 1 bucket)
        mock_any_8 = Any()
        mock_message_8 = PerceivedActorEvent(
            camera_uuid=mock_camera_uuid_1,
            timestamp_ms=base_start_time_ms - 900,
            end_timestamp_ms=base_start_time_ms + 7560000,
        )
        mock_any_8.Pack(mock_message_8)
        # Message 9: (camera_1, person time (after change), ~3.5 hours over three buckets)
        mock_any_9 = Any()
        mock_message_9 = PerceivedActorState(
            camera_uuid=mock_camera_uuid_1,
            timestamp_ms=base_start_time_ms - 7347123,
            end_timestamp_ms=base_start_time_ms + 4536473,
            actor_category=ActorCategory.ACTOR_CATEGORY_PERSON,
            person_lift_type=None,
            person_reach_type=False,
            run_uuid="v0.94.0:d45a7ce9-f5bb-4c32-abdc-c715adcb1463",
        )
        mock_any_9.Pack(mock_message_9)
        # Message 10: (camera_2, person time (before change) - invalid)
        mock_any_10 = Any()
        mock_message_10 = PerceivedActorState(
            camera_uuid=mock_camera_uuid_2,
            timestamp_ms=base_start_time_ms + 12000,
            end_timestamp_ms=base_start_time_ms + 500000,
            actor_category=ActorCategory.ACTOR_CATEGORY_PERSON,
            person_lift_type=True,
            person_reach_type=False,
            run_uuid="v0.93.0:d45a7ce9-f5bb-4c32-abdc-c715adcb1463",
        )
        mock_any_10.Pack(mock_message_10)
        # Message 11: (camera_2, person time (weird uuid) - valid)
        mock_any_11 = Any()
        mock_message_11 = PerceivedActorState(
            camera_uuid=mock_camera_uuid_2,
            timestamp_ms=base_start_time_ms,
            end_timestamp_ms=base_start_time_ms + 30000,
            actor_category=ActorCategory.ACTOR_CATEGORY_PERSON,
            person_lift_type=True,
            person_reach_type=False,
            run_uuid="v0.pizza.0:d45a7ce9-f5bb-4c32-abdc-c715adcb1463",
        )
        mock_any_11.Pack(mock_message_11)
        expected = {mock_camera_uuid_1: {}, mock_camera_uuid_2: {}}
        expected[mock_camera_uuid_1].update(
            {
                PerceivedActorStateDurationCategory.PIT_STATIONARY_TIME: {},
                PerceivedActorStateDurationCategory.PERSON_TIME: {},
            }
        )
        expected[mock_camera_uuid_1][
            PerceivedActorStateDurationCategory.PIT_STATIONARY_TIME
        ].update(
            {
                datetime(2023, 2, 7, 22): timedelta(seconds=93)
                + timedelta(seconds=67, microseconds=149000),
                datetime(2023, 2, 7, 23): timedelta(
                    minutes=4, seconds=56, microseconds=851000
                ),
            }
        )
        expected[mock_camera_uuid_1][
            PerceivedActorStateDurationCategory.PERSON_TIME
        ].update(
            {
                datetime(2023, 2, 7, 22): timedelta(
                    seconds=67, microseconds=149000
                )
                + timedelta(hours=1),
                datetime(2023, 2, 7, 23): timedelta(hours=1)
                + timedelta(hours=1),
                datetime(2023, 2, 8): timedelta(hours=1)
                + timedelta(seconds=813, microseconds=324000),
                datetime(2023, 2, 8, 1): timedelta(hours=1),
                datetime(2023, 2, 8, 2): timedelta(
                    minutes=39, seconds=56, microseconds=851000
                ),
                datetime(2023, 2, 7, 20): timedelta(
                    seconds=270, microseconds=272000
                ),
                datetime(2023, 2, 7, 21): timedelta(hours=1),
            }
        )
        expected[mock_camera_uuid_2].update(
            {
                PerceivedActorStateDurationCategory.PIT_STATIONARY_TIME: {},
                PerceivedActorStateDurationCategory.PIT_NON_STATIONARY_TIME: {},
                PerceivedActorStateDurationCategory.PERSON_TIME: {},
            }
        )
        expected[mock_camera_uuid_2][
            PerceivedActorStateDurationCategory.PIT_STATIONARY_TIME
        ].update(
            {
                datetime(2023, 2, 7, 22): timedelta(
                    seconds=67, microseconds=149000
                ),
                datetime(2023, 2, 7, 23): timedelta(hours=1),
                datetime(2023, 2, 8): timedelta(hours=1),
                datetime(2023, 2, 8, 1): timedelta(hours=1),
                datetime(2023, 2, 8, 2): timedelta(
                    minutes=39, seconds=56, microseconds=851000
                ),
            }
        )
        expected[mock_camera_uuid_2][
            PerceivedActorStateDurationCategory.PIT_NON_STATIONARY_TIME
        ].update(
            {
                datetime(2023, 2, 7, 22): timedelta(
                    seconds=124, microseconds=49000
                ),
                datetime(2023, 2, 7, 23): timedelta(hours=1),
                datetime(2023, 2, 8): timedelta(hours=1),
                datetime(2023, 2, 8, 1): timedelta(
                    minutes=3, seconds=56, microseconds=851000
                ),
            }
        )
        expected[mock_camera_uuid_2][
            PerceivedActorStateDurationCategory.PERSON_TIME
        ].update({datetime(2023, 2, 7, 22): timedelta(seconds=30)})
        raw_stream = io.BytesIO(
            base64.b64encode(mock_any_1.SerializeToString())
            + b"\n"
            + base64.b64encode(mock_any_2.SerializeToString())
            + b"\n"
            + base64.b64encode(mock_any_3.SerializeToString())
            + b"\n"
            + base64.b64encode(mock_any_4.SerializeToString())
            + b"\n"
            + base64.b64encode(mock_any_5.SerializeToString())
            + b"\n"
            + base64.b64encode(mock_any_6.SerializeToString())
            + b"\n"
            + base64.b64encode(mock_any_7.SerializeToString())
            + b"\n"
            + base64.b64encode(mock_any_8.SerializeToString())
            + b"\n"
            + base64.b64encode(mock_any_9.SerializeToString())
            + b"\n"
            + base64.b64encode(mock_any_10.SerializeToString())
            + b"\n"
            + base64.b64encode(mock_any_11.SerializeToString())
            + b"\n"
        )
        mock_batch = StreamingBody(raw_stream, len(raw_stream.getvalue()))
        obj = PerceivedActorStateDurationBatchProcessor()
        results = obj._process_batch(batch=mock_batch)
        self.assertEqual(expected, results)


if __name__ == "__main__":
    unittest.main()
