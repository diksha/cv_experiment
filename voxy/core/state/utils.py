#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

from google import api_core
from google.cloud import pubsub


def create_topic(topic_name):
    project_id = topic_name.split("/")[1]
    publish_client = pubsub.PublisherClient()
    project_path = f"projects/{project_id}"
    existing_topics = [
        topic.name
        for topic in publish_client.list_topics(
            request={"project": project_path}
        )
    ]

    if topic_name in existing_topics:
        return

    # Sometimes due to race condition when running multiple
    # graphs in parallel, it is the check for existing topics
    # returns None in multiple graphs and then they end up
    # here, causing failure for all except one.
    # Therefore still do a try/catch check.
    try:
        publish_client.create_topic(request={"name": topic_name})
    except api_core.exceptions.AlreadyExists:
        return


def create_subscription(topic_name, subscription_name):
    project_id = topic_name.split("/")[1]
    project_path = f"projects/{project_id}"
    subscriber_client = pubsub.SubscriberClient()
    with subscriber_client:
        existing_subscriptions = [
            subscription.name
            for subscription in subscriber_client.list_subscriptions(
                request={"project": project_path}
            )
        ]

    if subscription_name in existing_subscriptions:
        return

    # Sometimes due to race condition when running multiple
    # graphs in parallel, it is the check for existing subscribers
    # returns None in multiple graphs and then they end up
    # here, causing failure for all except one.
    # Therefore still do a try/catch check.
    try:
        subscriber_client = pubsub.SubscriberClient()
        with subscriber_client:
            subscriber_client.create_subscription(
                request={"name": subscription_name, "topic": topic_name}
            )
    except api_core.exceptions.AlreadyExists:
        return
