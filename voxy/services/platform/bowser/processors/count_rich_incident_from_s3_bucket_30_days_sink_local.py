from loguru import logger
from pyflink.common import Time
from pyflink.common.serialization import Encoder
from pyflink.datastream import RuntimeExecutionMode
from pyflink.datastream.connectors.file_system import FileSink
from pyflink.datastream.window import TumblingProcessingTimeWindows

from protos.platform.bowser.v1.bowser_config_keys_pb2 import Unit
from services.platform.bowser.engine.common.bowser_fondation import (
    BowserFactoryPipe,
)
from services.platform.bowser.engine.common.flatmaps.keys.incident_rich_key import (
    IncidentRichKey,
)
from services.platform.bowser.engine.common.flatmaps.state_or_event_to_incident_flat_map import (
    StateOrEventToIncidentFlatMap,
)
from services.platform.bowser.engine.utils.bowser_config_utils import (
    BowserConfigUtils,
)
from services.platform.bowser.engine.utils.bowser_proto_utils import (
    BowserProtoUtils,
)
from services.platform.bowser.engine.utils.bowser_utils import BowserUtils

# trunk-ignore-all(pylint/E0611): ignore pb import errors

PROCESSOR_NAME = "OFFICE DEPOT 1 MONTH FROM S3"

S3_CONSUMER = BowserProtoUtils.create_s3_consumer_proto(
    s3_bucket="voxel-perception-production-states-events",
    s3_bucket_uris=["data/office_depot/dallas/0001/cha/2023/03/01/"],
    consumer_name="CONSUME S3 STATES/EVENTS PROTO FILES",
)

KEYS = BowserProtoUtils.create_key_proto(
    tuple_key=("camera_uuid", "incident_type_id"),
    timestamp_mapping_key="start_frame_relative_ms",
    timestamp_unit_group_by=Unit.UNIT_HOURS,
)

LOCAL_SINK = BowserProtoUtils.create_sink_local_proto(
    sink_name="SINK TO LOCAL FILE",
    out_location="/tmp/bowser",  # trunk-ignore(bandit/B108)
)

WINDOW = BowserProtoUtils.create_window_proto(time_second=15)

LOG_LEVEL = "INFO"

if __name__ == "__main__":
    BowserUtils.try_set_bowser_home()
    BowserUtils.set_loger(LOG_LEVEL)
    args = BowserConfigUtils.parse_args("development")

    job_submit = (
        BowserFactoryPipe.new_raw_pipe(RuntimeExecutionMode.BATCH, 8)
        .consume_states_event_from_s3(
            S3_CONSUMER.aws.s3.buckets, S3_CONSUMER.name
        )
        .flat_map(
            StateOrEventToIncidentFlatMap(),
            "[FLATMAP] FIND INCIDENT OVER STATES/EVENTS",
        )
        .key_by_incident_with_window(
            key_reducer=IncidentRichKey(KEYS),
            window=TumblingProcessingTimeWindows.of(
                Time.seconds(WINDOW.time_second)
            ),
        )
        .count_incident_keys()
        .to_string()
        .sink_and_submit_job(
            (
                FileSink.for_row_format(
                    LOCAL_SINK.local.out_location,
                    Encoder.simple_string_encoder(),
                ).build()
            ),
            LOCAL_SINK.name,
        )
    )

    logger.info(f"Bowser Job : {job_submit.get_job_id()} is submit")
