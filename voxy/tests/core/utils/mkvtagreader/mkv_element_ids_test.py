import unittest

from core.utils.mkvtagreader import ElementId, mkv_element_ids


class MKVElementIdsTest(unittest.TestCase):
    def test_register(self):
        mkv_element_ids.register()
        self.assertIsNotNone(
            ElementId.by_name("Segment"),
            "mkv Segment type should be registered",
        )
