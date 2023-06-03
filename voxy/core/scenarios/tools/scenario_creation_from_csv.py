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
import os
from datetime import datetime

from core.scenarios.scenario_generator import (
    ScenarioIdentifier,
    ScenariosSource,
    ScenarioVideoGenerator,
)


class ScenariosSourceCSV(ScenariosSource):
    def __init__(self, csv_path, scenario_type="positive"):
        self.csv_path = csv_path
        self.scenario_type = scenario_type

    def extract_scenarios(self, incident_types_to_include=None):
        if not os.path.exists(self.csv_path):
            return []
        video_scenario_identifiers_map = {}
        with open(self.csv_path) as f:
            for line in f:
                line = self.validate_input(line)
                if line == "":
                    continue
                try:
                    (
                        incident_type,
                        start_timestamp,
                        scenario_duration,
                        video_url,
                    ) = line.split(",")
                except ValueError as e:
                    print("Skipped ", line, "due to error", e)
                    continue
                if (
                    incident_types_to_include is None
                    or incident_type in incident_types_to_include
                ):
                    video_uuid = video_url.replace(
                        "https://storage.cloud.google.com/voxel-external-activity-review/",
                        "",
                    )
                    video_uuid = video_uuid.replace(
                        "https://storage.cloud.google.com/voxel-raw-logs/",
                        "",
                    )

                    video_uuid = video_uuid.split(".")[0]
                    scenario_identifier = ScenarioIdentifier(
                        incident_type=incident_type,
                        start_timestamp=start_timestamp,
                        scenario_duration=scenario_duration,
                        scenario_type=self.scenario_type,
                    )
                    if video_uuid in video_scenario_identifiers_map.keys():
                        video_scenario_identifiers_map[video_uuid].append(
                            scenario_identifier
                        )
                    else:
                        video_scenario_identifiers_map[video_uuid] = [
                            scenario_identifier
                        ]
        return video_scenario_identifiers_map

    def sanitize_incident_type(self, incident_type):
        if "Hard Hat" in incident_type:
            return "HARD_HAT"
        if "Safety Vest" in incident_type:
            return "SAFETY_VEST"
        if "Bad Posture" in incident_type:
            return "BAD_POSTURE"
        if "No Stop At End Of Aisle" in incident_type:
            return "NO_STOP_AT_END_OF_AISLE"
        if "Over Reaching" in incident_type:
            return "OVERREACHING"

        return "unknown_incident"

    def sanitize_timestamps(self, timestamp):
        try:
            timestamp = ":".join(timestamp.split(":")[0:3])
            return timestamp
        except ValueError:
            return ""

    def validate_input(self, input):
        try:
            # ensure standard format
            input = ",".join(input.split(",")[0:4])
        except ValueError:
            return ""
        # ensure structure
        incident_type, start_timestamp, end_timestamp, video_url = input.split(
            ","
        )
        incident_type = self.sanitize_incident_type(incident_type)
        start_timestamp = self.sanitize_timestamps(start_timestamp)
        end_timestamp = self.sanitize_timestamps(end_timestamp)

        if (
            incident_type == "unknown_incident"
            or start_timestamp == ""
            or end_timestamp == ""
        ):
            return ""
        end_time = datetime.strptime(end_timestamp, "%H:%M:%S")
        start_time = datetime.strptime(start_timestamp, "%H:%M:%S")
        duration = str(end_time - start_time)
        input = ",".join([incident_type, start_timestamp, duration, video_url])
        return input


if __name__ == "__main__":
    scenario_source = ScenariosSourceCSV("/home/haddy/voxel/scenarios.csv")
    video_scenarios_map = scenario_source.extract_scenarios()
    scenario_generator = ScenarioVideoGenerator(buffer_s=5)
    uploaded_scenarios = []

    for video_uuid, scenario_identifiers in video_scenarios_map.items():
        print("Processing ", video_uuid, "....")
        generated_scenarios = scenario_generator.generate(
            video_uuid,
            scenario_identifiers,
            video_bucket="voxel-external-activity-review",
        )
        uploaded_scenarios = (
            uploaded_scenarios
            + scenario_generator.upload_to_gcs(generated_scenarios)
        )

    print("Uploaded Scenarios....")
    for uploaded_scenario in uploaded_scenarios:
        print(uploaded_scenario)
    print(
        "Please run the ingest script(https://buildkite.com/voxel/create-labeling-tasks) to create labeling tasks for these scenarios"
    )
