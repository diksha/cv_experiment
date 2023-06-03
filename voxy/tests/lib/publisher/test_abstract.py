import unittest
from typing import Tuple
from unittest.mock import Mock, call, patch

from lib.publisher.abstract import AbstractPublisher, PublishingError

# trunk-ignore-all(pylint/W0212)


def get_test_objects() -> Tuple[Mock, Mock, AbstractPublisher]:
    """Helper for setting up tests

    Returns:
        Tuple[Mock, Mock, AbstractPublisher]: error callback, success callback and publisher client
    """
    error_callback, success_callback = Mock(), Mock()
    return (
        error_callback,
        success_callback,
        AbstractPublisher(10, error_callback, success_callback),
    )


class AbstractPublisherTest(unittest.TestCase):
    def test_successful_publish(self):
        error_callback, success_callback, publisher = get_test_objects()
        with (
            patch.object(
                AbstractPublisher, "_publish_record", return_value="success"
            ) as mock_publish_record,
            patch.object(
                AbstractPublisher, "_handle_response"
            ) as mock_handle_response,
        ):
            publisher._put_record("record")
            publisher.close()

            mock_publish_record.assert_called_once_with("record", None)
            success_callback.assert_called_once_with("success")
            error_callback.assert_not_called()

            mock_handle_response.assert_called_once_with("success")

    def test_failed_publish(self):
        error_callback, success_callback, publisher = get_test_objects()
        error = PublishingError("error occurred")
        with (
            patch.object(
                AbstractPublisher,
                "_publish_record",
                side_effect=error,
            ) as mock_publish_record,
        ):
            publisher._put_record("record", metadata=None)
            publisher.close()

            mock_publish_record.assert_called_once_with("record", None)
            success_callback.assert_not_called()
            error_callback.assert_called_once_with(error)

    def test_publish_with_metadata(self):
        _, _, publisher = get_test_objects()
        input_record = ("record", {"meta": "data"})
        with (
            patch.object(
                AbstractPublisher, "_publish_record", return_value="success"
            ) as mock_publish_record,
            patch.object(AbstractPublisher, "_handle_response"),
        ):
            publisher._put_record(input_record[0], input_record[1])
            publisher.close()

            mock_publish_record.assert_called_once_with(
                input_record[0], input_record[1]
            )

    def test_publishes_in_order(self):
        _, _, publisher = get_test_objects()
        input_records = range(10)

        with (
            patch.object(
                AbstractPublisher, "_publish_record", return_value="success"
            ) as mock_publish_record,
            patch.object(AbstractPublisher, "_handle_response"),
        ):
            for record in input_records:
                publisher._put_record(record)

            publisher.close()

            assert mock_publish_record.call_args_list == list(
                map(lambda x: call(x, None), input_records)
            )
