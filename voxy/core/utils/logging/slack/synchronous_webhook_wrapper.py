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

import typing

from slack_sdk.webhook import WebhookClient


class SynchronousWebhookWrapper:
    """
    A simple wrapper for a slack webhook. Supports
    connecting to different channels and sending messages
    """

    def __init__(self, webhook_client: WebhookClient):
        self.webhook = webhook_client

    def post_message(self, message: str) -> None:
        """
        Post a message to the perception verbose channel
        """
        response = self.webhook.send(text=message)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to send message {message}")

    def post_message_block(self, blocks: list) -> None:
        """
        Post a message block to the perception verbose channel. See https://api.slack.com/block-kit/building#getting_started for more details
        """
        response = self.webhook.send(blocks=blocks)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to send block, error, {response.body}, block, {blocks}"
            )

    def post_message_block_with_fields(
        self, title: str, blocks: typing.List[str]
    ):
        """
        Posts the message with a title and a set of message blocks.
        Truncates messages based on the recommendations from
        https://api.slack.com/reference/block-kit/blocks

        Args:
            title (str): the title for the message block
            blocks (typing.List[str]): the set of fields
        """

        def truncate_to(item: str, n_characters: int) -> str:
            """
            Truncates a string to n_characters with an ellipsis

            Args:
                item (str): the string to truncate
                n_characters (int): the number of characters

            Returns:
                str: the truncated string
            """
            if len(item) < n_characters:
                return item
            if n_characters - 3 < 1:
                return "..."
            return item[: (n_characters - 3)] + "..."

        # see: https://api.slack.com/reference/block-kit/blocks
        # for more information on the max section length
        # truncate if necessary
        notification_block = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": truncate_to(title, 100),
                },
            }
        ]
        link_fields = []
        for item in blocks:
            link_fields.append(
                {
                    "type": "mrkdwn",
                    "text": truncate_to(item, 2000),
                }
            )
        step = 9
        for index in range(0, len(link_fields), step):
            sub_link_fields = link_fields[index : index + step]
            notification_block.append(
                {
                    "type": "section",
                    "fields": sub_link_fields,
                }
            )
        self.post_message_block(notification_block)
