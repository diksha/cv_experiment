import base64
from typing import Iterable, Tuple, Union

from google.protobuf.any_pb2 import Any
from loguru import logger
from pyflink.datastream import FlatMapFunction, RuntimeContext

from core.structs.event import Event

# trunk-ignore(pylint/E0611)
from core.structs.protobufs.v1.event_pb2 import Event as EventPb

# trunk-ignore(pylint/E0611)
from core.structs.protobufs.v1.state_pb2 import State as StatePb
from core.structs.state import State


class AwsS3ObjectToStatesEventsFlatMap(FlatMapFunction):
    def __init__(self, client):

        self.client = None
        self._aws_client = client

    def open(self, runtime_context: RuntimeContext):
        """Initializes s3 client

        Args:
            runtime_context (RuntimeContext): bowser runtime context
        """
        self.client = self._aws_client.client("s3")

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
            state or event or null messages
        """

        response = self.client.get_object(
            Bucket=key[0],
            Key=key[1],
        )
        for line in response["Body"].iter_lines():
            message = self.decode(line)
            if message is not None:
                yield message
