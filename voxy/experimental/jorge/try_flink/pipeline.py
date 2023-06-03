import argparse
import base64
import logging
import os
import sys
from typing import Iterable, Tuple, Union

import boto3
import yaml
from google.protobuf.any_pb2 import Any
from pyflink.common import Types
from pyflink.datastream import (
    FlatMapFunction,
    RuntimeContext,
    RuntimeExecutionMode,
    StreamExecutionEnvironment,
)
from rules_python.python.runfiles import runfiles

from core.execution.nodes.incident_machine import IncidentMachineNode
from core.structs.event import Event

# trunk-ignore(pylint/E0611)
from core.structs.protobufs.v1.event_pb2 import Event as EventPb

# trunk-ignore(pylint/E0611)
from core.structs.protobufs.v1.state_pb2 import State as StatePb
from core.structs.state import State

logger = logging.getLogger(__name__)


AWS_SESSION = None


def _aws_session():
    # trunk-ignore(pylint/W0603): useful global
    global AWS_SESSION
    if AWS_SESSION is None:
        if os.getenv("ENVIRONMENT") == "production":
            sts = boto3.client("sts")
            response = sts.assume_role(
                RoleArn="arn:aws:iam::360054435465:role/flink_testing_access_assumable_role",
                RoleSessionName="experimental_jorge_try_flink_production",
            )
            AWS_SESSION = boto3.Session(
                aws_access_key_id=response["Credentials"]["AccessKeyId"],
                aws_secret_access_key=response["Credentials"][
                    "SecretAccessKey"
                ],
                aws_session_token=response["Credentials"]["SessionToken"],
            )
        else:
            AWS_SESSION = boto3.Session()

    return AWS_SESSION


class GetSortedObjectKeysFromS3(FlatMapFunction):
    def __init__(self):
        self.client = None

    def open(self, runtime_context: RuntimeContext) -> None:
        """Initializes the boto3 client

        Args:
            runtime_context (str): flink runtime context
        """
        self.client = _aws_session().client("s3")

    def flat_map(self, bucket: str) -> Iterable[Tuple[str, str]]:
        """Receives a bucket and produces an iterable of s3 objects

        Args:
            bucket (str): _description_

        Yields:
            _type_: _description_
        """
        client = _aws_session().client("s3")

        response = client.list_objects_v2(
            Bucket=bucket,
            Prefix="data",
        )

        keys = [o["Key"] for o in response["Contents"]]

        while response["IsTruncated"]:
            response = client.list_objects_v2(
                Bucket=bucket,
                Prefix="data",
                ContinuationToken=response["NextContinuationToken"],
            )

            keys.extend([o["Key"] for o in response["Contents"]])

        keys.sort(key=lambda x: "-".join(x.split("/")[-1].split("-")[4:-5]))

        for k in keys:
            yield (bucket, k)


class GetFramesFromObject(FlatMapFunction):
    def __init__(self):
        self.client = None

    def open(self, runtime_context: RuntimeContext):
        """Initializes s3 client

        Args:
            runtime_context (RuntimeContext): flink runtime context
        """
        self.client = _aws_session().client("s3")

    def decode(self, line: str) -> Union[State, Event, None]:
        """Decodes a base64 encoded protobuf Any message into a StatePb or EventPb

        Args:
            line (str): base64 encoded binary protobuf

        Returns:
            Union[StatePb, EventPb, None]: encoded message result, None if unsupported

        Raises:
            Exception: there was an error decoding the message
        """
        anypb = Any.FromString(base64.b64decode(line))
        if anypb.Is(StatePb.DESCRIPTOR):
            statepb = StatePb()
            anypb.Unpack(statepb)
            try:
                return State.from_proto(statepb)
            except Exception as ex:
                print(f"failed on statepb line: {ex}")
                raise
        elif anypb.Is(EventPb.DESCRIPTOR):
            eventpb = EventPb()
            anypb.Unpack(eventpb)
            try:
                return Event.from_proto(eventpb)
            except Exception as ex:
                print(f"failed on eventpb line: {ex}")
                raise
        else:
            logger.warning("invalid protobuf message %s", anypb)
            return None

    def flat_map(
        self, key: Tuple[str, str]
    ) -> Iterable[Union[State, Event, None]]:
        """Receives s3 objects and produces an iterable of state or event messages

        Args:
            key (Tuple[str, str]): tuple of s3 objects

        Yields:
            Iterable: state/event messages
        """
        response = self.client.get_object(
            Bucket=key[0],
            Key=key[1],
        )
        for line in response["Body"].iter_lines():
            message = self.decode(line)
            if message is not None:
                yield message


class RunIncidentMachine(FlatMapFunction):
    def __init__(self):
        self._incident_machine = None

    def _load_incident_machine(self, camera_uuid):
        """Attempts to load up the relevant incident machine for a camera

        Args:
            camera_uuid (str): camera uuid to load config for
        """
        if self._incident_machine is not None:
            return

        runf = runfiles.Create()
        with open(
            runf.Rlocation(f"voxel/configs/cameras/{camera_uuid}.yaml"),
            mode="r",
            encoding="utf-8",
        ) as gcf:
            graph_config = yaml.safe_load(gcf)

        self._incident_machine = IncidentMachineNode(graph_config)

    def flat_map(self, state_or_event):
        """Consumes state/event messages and runs them through an incident machine

        Args:
            state_or_event (Union[StatePb, EventPb]): _description_

        Yields:
            _type_: _description_
        """
        self._load_incident_machine(state_or_event.camera_uuid)

        if self._incident_machine is not None:
            incidents = self._incident_machine.process([state_or_event])
            for inc in incidents:
                yield inc


def process_state_event_messages():
    """Runs a flink pipeline to process state/event messages from S3"""
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)

    buckets = env.from_collection(
        ["voxel-perception-production-states-events"]
    ).name("bucket name")
    object_keys = (
        buckets.flat_map(GetSortedObjectKeysFromS3())
        .name("get s3 objects")
        .key_by(lambda x: "/".join(x[1].split("/")[1:5]))
    )
    states_events = object_keys.flat_map(GetFramesFromObject()).name(
        "get states events"
    )
    keyed_states_events = states_events.key_by(
        lambda x: x.camera_uuid, key_type=Types.STRING()
    )

    incidents = keyed_states_events.flat_map(RunIncidentMachine()).name(
        "run incident machine"
    )

    incidents.print()

    env.execute()


def _try_set_flink_home():
    if os.getenv("FLINK_HOME") is not None:
        return

    try:
        runfiles_dir = runfiles.Create().EnvVars()["RUNFILES_DIR"]
        flink_libraries = os.path.join(
            runfiles_dir,
            "pip_deps_apache_flink_libraries/site-packages/pyflink",
        )

        if not os.path.exists(flink_libraries):
            logger.warning("could not find flink libraries")
            return

        os.environ["FLINK_LIB_DIR"] = os.path.join(flink_libraries, "lib")
        os.environ["FLINK_OPT_DIR"] = os.path.join(flink_libraries, "opt")
        os.environ["FLINK_PLUGINS_DIR"] = os.path.join(
            flink_libraries, "plugins"
        )

    # trunk-ignore(pylint/W0703): this is best effort and should fail with only a warning
    except Exception as ex:
        logger.warning("failed to set flink env vars: %s", ex)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format="%(message)s"
    )

    _try_set_flink_home()

    parser = argparse.ArgumentParser()
    argv = sys.argv[1:]
    known_args, _ = parser.parse_known_args(argv)

    process_state_event_messages()
