import io
import unittest

from rules_python.python.runfiles import runfiles

from core.utils.mkvtagreader import MKVTagReader


def _load_testdata(filename) -> bytes:
    with open(runfiles.Create().Rlocation(filename), "rb") as f:
        return f.read()


class MKVTagReaderTest(unittest.TestCase):
    def test_open_testdata(self):
        testdata = _load_testdata("mkvtagreader_testdata/testfile.mkv")
        self.assertGreater(
            len(testdata), 0, "content length is greater than zero"
        )

    def test_read_mkv(self):
        testinput = _load_testdata("mkvtagreader_testdata/testfile.mkv")
        tagr = MKVTagReader(io.BytesIO(testinput))
        testoutput = tagr.read(-1)
        self.assertEqual(testinput, testoutput, "input and output are equal")
        self.assertIn(
            "AWS_KINESISVIDEO_CONTINUATION_TOKEN",
            tagr.tags(),
            "can find tags",
        )

    def test_unknown_len_cluster(self):
        testinput = _load_testdata(
            "mkvtagreader_testdata/unknown_len_cluster.mkv"
        )
        tagr = MKVTagReader(io.BytesIO(testinput))
        testoutput = tagr.read(-1)
        self.assertEqual(testinput, testoutput, "input and output match")
