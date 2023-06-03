import io
import os
import unittest
from contextlib import ExitStack

from rules_python.python.runfiles import runfiles

from core.utils.pyav_decoder import PyavDecoder

MP4_TEST_FILE = "artifacts_office_cam_mp4/office_cam.mp4"


def _mp4_test_file_path():
    """Returns correct path to runfiles

    Returns:
        str: test file path
    """
    runf = runfiles.Create()
    return runf.Rlocation(MP4_TEST_FILE)


class PyavDecoderTest(unittest.TestCase):
    def test_have_debug_video(self):
        self.assertTrue(os.path.exists(_mp4_test_file_path()))

    def assertDecodeOneFrame(self, decoder):
        frame = decoder.get_frame()
        self.assertIsNotNone(frame, "frame should not be none")
        self.assertEqual(
            decoder.get_width(), frame.frame.width, "width should match"
        )
        self.assertEqual(
            decoder.get_height(),
            frame.frame.height,
            "height should match",
        )
        self.assertEqual(
            decoder.get_last_timestamp_ms(),
            frame.timestamp_ms(),
            "last timestamp should match",
        )

    def test_decode_video_filepath(self):
        with ExitStack() as stack:
            decoder = PyavDecoder(_mp4_test_file_path(), 5)
            stack.callback(decoder.close)
            self.assertDecodeOneFrame(decoder)

    def test_decode_video_binary_data(self):
        mp4_data = None
        with open(_mp4_test_file_path(), "rb") as mp4file:
            mp4_data = mp4file.read(-1)

        with ExitStack() as stack:
            decoder = PyavDecoder(io.BytesIO(mp4_data), fps=5)
            stack.callback(decoder.close)
            self.assertDecodeOneFrame(decoder)

    def test_decode_video_framerate(self):
        target_fps = 5
        with ExitStack() as stack:
            decoder = PyavDecoder(_mp4_test_file_path(), 5)
            stack.callback(decoder.close)

            start_ts, end_ts = None, None
            frame_count = 0

            frame = decoder.get_frame()
            while frame is not None:
                if start_ts is None:
                    start_ts = frame.timestamp_ms()
                end_ts = frame.timestamp_ms()
                frame_count += 1

                frame = decoder.get_frame()
                if end_ts - start_ts > 5000:
                    # only process 5s of data so this test isn't too slow
                    break

            self.assertIsNotNone(
                start_ts, "failed to get start_ts, video is likely invalid"
            )
            self.assertIsNotNone(
                end_ts, "failed to get end_ts, video is likely invalid"
            )
            actual_fps = frame_count / (end_ts - start_ts) * 1000

            # ensure we are with 10% of the target fps
            self.assertGreater(
                actual_fps,
                0.9 * target_fps,
                f"fps should be greater than {0.9*target_fps}",
            )
            self.assertLess(
                actual_fps,
                1.1 * target_fps,
                f"fps should be less than {1.1*target_fps}",
            )

    def test_decoder_iterator_and_context(self):
        with PyavDecoder(_mp4_test_file_path(), 5) as decoder:
            count = 0
            for frame in decoder:
                self.assertEqual(
                    frame.timestamp_ms(),
                    decoder.get_last_timestamp_ms(),
                    "frame timesatmp and last_timestamp_ms should match",
                )
                count += 1
                if count > 5:
                    return

    def test_decoder_no_fps(self):
        with PyavDecoder(_mp4_test_file_path()) as decoder:
            start_ts, end_ts = None, None
            count = 0
            for frame in decoder:
                if start_ts is None:
                    start_ts = frame.timestamp_ms()
                end_ts = frame.timestamp_ms()

                # count the number of frames in the first second
                if end_ts - start_ts <= 1000:
                    count += 1
                else:
                    break

            # test content has 21 frames in the first second, this value
            # will need to be updated if we swap out the test content
            self.assertEqual(
                count,
                21,
                "number of frames in the first second of test content should be correct",
            )
