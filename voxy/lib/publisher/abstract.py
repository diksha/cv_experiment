import atexit
from queue import Queue
from threading import Thread
from typing import Any, Callable, Optional


class PublishingError(Exception):
    pass


class AbstractPublisher:
    """Interface for publishing data to a specific stream and partition in Kinesis Data Streams"""

    def __init__(
        self,
        max_buffer_size: int,
        error_callback: Callable[[PublishingError], Any] = None,
        success_callback: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Args:
            max_buffer_size (int):
                The maximum number of records to store in memory before dropping additional records
            error_callback (Optional[Callable[[PublishingError], Any]]):
                Callback function to handle exceptions that cause dropped data
                Defaults to throwing error
            success_callback (Optional[Callable[[Dict[str, str]], Any]], optional):
                Callback function that takes the response of a successful publish
                Defaults to None.
        """

        self._max_buffer_size = max_buffer_size
        self._error_callback = error_callback
        self._success_callback = success_callback

        self._record_queue = Queue()

        self._publishing_thread = Thread(target=self._publish)
        self._publishing_thread.start()

        atexit.register(self.close)

    def _put_record(self, record: Any, metadata: Any = None):
        """Adds record to the queue of records to be published

        Derived publishers should wrap this call and not expose
        the metadata parameter to public API

        Raises:
            ValueError: if record is None

        Args:
            record (Any): record to be buffered and then published
            metadata (Any, optional):
                Anything attached to the record that should not be published but may be
                required for publishing
        """
        if record is None:
            raise ValueError("record must not be None")

        if self._record_queue.qsize() < self._max_buffer_size:
            self._record_queue.put((record, metadata))
        else:
            err = PublishingError(
                BufferError("failed to put record to full buffer")
            )
            # Correct API usage so allow user to handle error in callback
            if self._error_callback is not None:
                self._error_callback(err)
            else:
                raise err

    def close(self):
        """Waits for all data to be pushed and for publishing thread to exit"""
        # unblock the get call in in _publish
        self._record_queue.put(None, block=True)
        self._publishing_thread.join()

    def _publish(self):
        """Continuously calls PutRecord while there are records queued"""
        while True:
            record = self._record_queue.get(block=True, timeout=None)

            # exit case
            if record is None:
                return

            data, metadata = record

            try:
                res = self._publish_record(data, metadata)
                self._handle_response(res)

                if self._success_callback is not None:
                    self._success_callback(res)
            except PublishingError as err:
                if self._error_callback is not None:
                    self._error_callback(err)
                else:
                    raise err

    def _publish_record(self, data: Any, metadata: Any):
        raise NotImplementedError(
            "_publish_record must be defined by derived class"
        )

    def _handle_response(self, res):
        raise NotImplementedError(
            "_handle_response must be defined by derived class"
        )
