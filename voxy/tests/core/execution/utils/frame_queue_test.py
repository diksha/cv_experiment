import unittest

import numpy

from core.execution.utils.frame_queue import FrameQueue


class FrameQueueTest(unittest.TestCase):
    def test_put_get(self):
        fq = FrameQueue(max_length=10)
        frame, pts = fq.get()
        self.assertIsNone(
            frame, "frame returned from get should be none on an empty queue"
        )
        frame, pts = numpy.ndarray((0,)), 1
        fq.put(frame, pts)
        self.assertEqual(
            len(fq),
            1,
            "should be 1 frame in the queue after a put with no gets",
        )

        retframe, retpts = fq.get()
        self.assertEqual(
            len(fq),
            0,
            "queue should be empty again after a single put followed by a single get",
        )
        self.assertIs(
            frame,
            retframe,
            "frame from get should be the same frame passed to put",
        )
        self.assertEqual(
            pts,
            retpts,
            "pts for frame from get should match pts passed to put",
        )

    def test_max_len(self):
        fq = FrameQueue(max_length=1)
        fq.put(numpy.ndarray((0,)), 1)
        fq.put(numpy.ndarray((0,)), 2)
        self.assertEqual(
            len(fq), 1, "frame queue length should be capped to 1"
        )
