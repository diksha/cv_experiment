import tempfile
import time
import unittest

from core.utils.publisher.publisher import Publisher


class IncidentPublisherTest(unittest.TestCase):
    def test_incident_json_filepaths_sorted_by_mtime_ascending(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # Arrange
            expected_files = [
                f"{tempdir}/c",
                f"{tempdir}/b",
                f"{tempdir}/a",
                f"{tempdir}/z",
                f"{tempdir}/aa",
                f"{tempdir}/_",
            ]

            for file in expected_files:
                # trunk-ignore(pylint/R1732): file created within tempdir
                open(file, "w", encoding="utf-8").close()

                # If we don't sleep, mtime will be the same for all files
                time.sleep(0.01)

            # Act
            publisher = Publisher()
            filepaths = publisher.get_incident_json_filepaths(f"{tempdir}/*")

            # Assert
            assert list(filepaths) == expected_files
