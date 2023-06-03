import pytest
from google.cloud.pubsub_v1.subscriber.message import Message

from core.portal.state.common.callbacks import (
    event_message_callback,
    state_message_callback,
)
from core.structs.protobufs.v1.event_pb2 import Event as StatePb
from core.structs.protobufs.v1.state_pb2 import State as EventPb


def get_state_message():
    """Get a state proto message.

    Returns:
        StatePb: state proto message
    """
    state = StatePb()
    state.camera_uuid = "sandbox/place/zone/cha"
    state.timestamp_ms = 0
    return state


def get_event_message():
    """Get an event message.

    Returns:
        EventPb: event proto message
    """
    event = EventPb()
    event.camera_uuid = "sandbox/place/zone/cha"
    event.timestamp_ms = 0
    return event


@pytest.mark.django_db
def test_state_callback() -> None:
    """Test state message callback happy path doesn't throw exception."""
    message = Message
    message.data = get_state_message().SerializeToString()
    message.ack = lambda: None
    print(message.data)
    state_message_callback(message)


@pytest.mark.django_db
def test_event_callback() -> None:
    """Test event message callback happy path doesn't throw exception."""
    message = Message
    message.data = get_event_message().SerializeToString()
    message.ack = lambda: None
    print(message.data)
    event_message_callback(message)
