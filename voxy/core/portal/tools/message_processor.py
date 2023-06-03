# trunk-ignore-all(pylint/C0413,flake8/E402)
import argparse
import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.portal.voxel.settings")
django.setup()


from django.conf import settings

from core.portal.state.common.callbacks import (
    event_message_callback,
    state_message_callback,
)
from core.portal.state.common.streaming_pull_handler import (
    StreamingPullHandler,
)
from core.state.utils import create_subscription, create_topic


class MessageProcessor:

    registry_topic_callback = {
        "event": [settings.PUB_SUB_EVENT_TOPIC, event_message_callback],
        "state": [settings.PUB_SUB_STATE_TOPIC, state_message_callback],
    }

    def __init__(self, message_type: str):
        if settings.PUBSUB_EMULATOR_HOST:
            self.setup_for_local_development(message_type)

        topic, callback = self.registry_topic_callback[message_type]
        self.streaming_pull_handler = StreamingPullHandler(
            subscription_path=f"{topic.replace('topics', 'subscriptions')}-subscription",
            callback=callback,
        )

    def run(self) -> None:
        """Run the message processor."""
        self.streaming_pull_handler.pull()

    def setup_for_local_development(self, message_type: str):
        """Set up dependencies for local development.

        Args:
            message_type (str): message type
        """
        topic, _ = self.registry_topic_callback[message_type]
        create_topic(topic)
        create_subscription(
            topic,
            f"{topic.replace('topics', 'subscriptions')}-subscription",
        )


def parse_args():
    """Parse command line args.

    Returns:
        Namespace: parsed args namespace (dict)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--message_type",
        "-e",
        type=str,
        choices=["state", "event"],
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    MessageProcessor(parsed_args.message_type).run()
