import unittest
from datetime import datetime, timedelta
from uuid import uuid4

import pytz
from django.test import TestCase as DjangoTestCase

from core.portal.api.models.incident import Incident, ReviewStatus
from core.portal.api.models.incident_type import (
    CameraIncidentType,
    IncidentType,
)
from core.portal.perceived_data.calculation_methods import (
    PerceivedEventRateCalculationMethodMapping,
)
from core.portal.perceived_data.enums import (
    PerceivedActorStateDurationCategory,
    PerceivedEventRateCalculationMethod,
)
from core.portal.perceived_data.models.perceived_actor_state_duration_aggregate import (
    PerceivedActorStateDurationAggregate,
    TimeBucketWidth,
)
from core.portal.perceived_data.models.perceived_event_rate_hourly import (
    Camera,
    PerceivedEventRateDefinition,
    PerceivedEventRateHourly,
)
from core.portal.perceived_data.query_definition_helpers import (
    query_for_camera_calculation_definitions,
)


# trunk-ignore-all(pylint/C0301)
class TestPerceivedEventRateCalculationMethods(DjangoTestCase):
    # trunk-ignore(pylint/C0103)
    def setUp(self):
        """Sets-up a DB to be used in testing"""
        incident_type_1 = IncidentType.objects.create(
            key="test_incident_type_1",
            name="test incident type 1",
            value="1001",
        )
        incident_type_2 = IncidentType.objects.create(
            key="test_incident_type_2",
            name="test incident type 2",
            value="1002",
        )
        incident_type_3 = IncidentType.objects.create(
            key="test_incident_type_3",
            name="test incident type 3",
            value="1003",
        )
        # Unrelated
        IncidentType.objects.create(
            key="test_incident_type_4",
            name="test incident type 4",
            value="1004",
        )

        self.def_a_1 = PerceivedEventRateDefinition.objects.create(
            incident_type=incident_type_1,
            name="def a1",
            calculation_method=1,
            perceived_actor_state_duration_category=PerceivedActorStateDurationCategory.PERSON_TIME,
        )
        self.def_b_1 = PerceivedEventRateDefinition.objects.create(
            incident_type=self.def_a_1.incident_type,
            name="def b1",
            calculation_method=1,
            perceived_actor_state_duration_category=PerceivedActorStateDurationCategory.PIT_STATIONARY_TIME,
        )
        self.def_c_1 = PerceivedEventRateDefinition.objects.create(
            incident_type=incident_type_2,
            name="def c1",
            calculation_method=1,
            perceived_actor_state_duration_category=self.def_b_1.perceived_actor_state_duration_category,
        )
        self.def_a_2 = PerceivedEventRateDefinition.objects.create(
            incident_type=self.def_a_1.incident_type,
            name="def a2",
            calculation_method=2,
            perceived_actor_state_duration_category=self.def_a_1.perceived_actor_state_duration_category,
        )
        self.def_b_2 = PerceivedEventRateDefinition.objects.create(
            incident_type=self.def_b_1.incident_type,
            name="def b2",
            calculation_method=2,
            perceived_actor_state_duration_category=self.def_b_1.perceived_actor_state_duration_category,
        )
        self.def_c_2 = PerceivedEventRateDefinition.objects.create(
            incident_type=self.def_c_1.incident_type,
            name="def c2",
            calculation_method=2,
            perceived_actor_state_duration_category=self.def_c_1.perceived_actor_state_duration_category,
        )
        # Unrelated
        PerceivedEventRateDefinition.objects.create(
            incident_type=incident_type_3,
            name="def 4",
            calculation_method=1,
            perceived_actor_state_duration_category=PerceivedActorStateDurationCategory.PIT_STATIONARY_TIME,
        )
        PerceivedEventRateDefinition.objects.create(
            incident_type=incident_type_2,
            name="def 5",
            calculation_method=2,
            perceived_actor_state_duration_category=PerceivedActorStateDurationCategory.PIT_NON_STATIONARY_TIME,
        )

        self.camera_a = Camera.objects.create(uuid="self.camera_a")
        self.camera_b = Camera.objects.create(uuid="self.camera_b")
        self.camera_c = Camera.objects.create(uuid="camera_c")
        # Unrelated
        Camera.objects.create(uuid="camera_d")
        CameraIncidentType.objects.get_or_create(
            camera_id=self.camera_a.id,
            incident_type=self.def_a_1.incident_type,
        )
        CameraIncidentType.objects.get_or_create(
            camera_id=self.camera_a.id,
            incident_type=self.def_b_2.incident_type,
        )
        CameraIncidentType.objects.get_or_create(
            camera_id=self.camera_b.id,
            incident_type=self.def_b_2.incident_type,
        )
        CameraIncidentType.objects.get_or_create(
            camera_id=self.camera_b.id,
            incident_type=self.def_c_1.incident_type,
        )
        # camera_id_to_def_ids = {
        #     self.camera_a.id: {
        #         self.def_a_2.id,
        #         self.def_b_2.id,
        #     },
        #     self.camera_b.id: {self.def_b_2.id, self.def_c_2.id},
        #     self.camera_c.id: {self.def_c_2.id},
        # }

        self.start_time = datetime(2023, 5, 11, 10, tzinfo=pytz.UTC)
        self.start_time_ms = self.start_time.timestamp() * 1000
        self.end_time = self.start_time + timedelta(hours=4)
        self.end_time_ms = self.end_time.timestamp() * 1000
        # Unrelated
        Incident.objects.create(
            uuid=uuid4(),
            title="test",
            incident_type=self.def_a_2.incident_type,
            camera=self.camera_a,
            timestamp=self.start_time - timedelta(seconds=1),
            review_status=ReviewStatus.VALID,
            valid_feedback_count=3,
            data={
                "uuid": "6543a8b7-ce82-4a3b-9aeb-d73fc8d019ee",
                "title": "Open Door duration",
                "priority": "medium",
                "actor_ids": "0",
                "camera_uuid": "americold/modesto/0003/cha",
                "video_s3_path": "s3://voxel-portal-production/americold/modesto/incidents/6543a8b7-ce82-4a3b-9aeb-d73fc8d019ee_video.mp4",
                "video_gcs_path": "gs://voxel-portal/incidents/6543a8b7-ce82-4a3b-9aeb-d73fc8d019ee_video.mp4",
                "docker_image_tag": "v0.28.0",
                "incident_type_id": "OPEN_DOOR_DURATION",
                "incident_version": "1.0",
                "organization_key": "AMERICOLD",
                "post_end_buffer_ms": "6000",
                "annotations_s3_path": "s3://voxel-portal-production/americold/modesto/incidents/6543a8b7-ce82-4a3b-9aeb-d73fc8d019ee_annotations.json",
                "pre_start_buffer_ms": "6000",
                "annotations_gcs_path": "gs://voxel-portal/incidents/6543a8b7-ce82-4a3b-9aeb-d73fc8d019ee_annotations.json",
                "camera_config_version": "2",
                "end_frame_relative_ms": "1676607497349",
                "start_frame_relative_ms": "894880800000",
                "video_thumbnail_s3_path": "s3://voxel-portal-production/americold/modesto/incidents/6543a8b7-ce82-4a3b-9aeb-d73fc8d019ee_thumbnail.jpg",
                "video_thumbnail_gcs_path": "gs://voxel-portal/incidents/6543a8b7-ce82-4a3b-9aeb-d73fc8d019ee_thumbnail.jpg",
            },
        )
        Incident.objects.create(
            uuid=uuid4(),
            title="test",
            incident_type=self.def_a_1.incident_type,
            camera=self.camera_a,
            timestamp=self.start_time + timedelta(seconds=1),
            # Don't show this
            experimental=True,
            review_status=ReviewStatus.VALID,
            valid_feedback_count=3,
            data={
                "uuid": "123",
                "start_frame_relative_ms": f"{self.start_time_ms}",
                "end_frame_relative_ms": f"{self.start_time_ms + (timedelta(minutes=3).total_seconds() * 1000)}",
                "incident_type_id": "OPEN_DOOR_DURATION",
            },
        )
        Incident.objects.create(
            uuid=uuid4(),
            title="test",
            incident_type=self.def_a_1.incident_type,
            camera=self.camera_a,
            timestamp=self.start_time + timedelta(minutes=2),
            review_status=ReviewStatus.VALID,
            valid_feedback_count=3,
            data={
                "uuid": "123",
                "start_frame_relative_ms": f"{self.start_time_ms}",
                "end_frame_relative_ms": f"{self.start_time_ms + (timedelta(minutes=3).total_seconds() * 1000)}",
                "incident_type_id": "OPEN_DOOR_DURATION",
            },
        )
        Incident.objects.create(
            uuid=uuid4(),
            title="test",
            incident_type=self.def_a_1.incident_type,
            camera=self.camera_a,
            timestamp=self.start_time + timedelta(minutes=12, seconds=59),
            review_status=ReviewStatus.VALID,
            valid_feedback_count=3,
            data={
                "uuid": "1ds",
                "start_frame_relative_ms": f"{self.start_time_ms}",
                "incident_type_id": "OPEN_DOOR_DURATION",
            },
        )
        Incident.objects.create(
            uuid=uuid4(),
            title="test",
            incident_type=self.def_a_1.incident_type,
            camera=self.camera_a,
            timestamp=self.start_time + timedelta(minutes=20, seconds=59),
            review_status=ReviewStatus.VALID,
            valid_feedback_count=3,
            data={
                "uuid": "123dd",
                "end_frame_relative_ms": f"{self.end_time_ms + 303030}",
                "incident_type_id": "OPEN_DOOR_DURATION",
            },
        )
        Incident.objects.create(
            uuid=uuid4(),
            title="test",
            incident_type=self.def_a_1.incident_type,
            camera=self.camera_a,
            timestamp=self.start_time + timedelta(seconds=1),
            review_status=ReviewStatus.VALID,
            valid_feedback_count=3,
            data={
                "uuid": "1ee23",
                "start_frame_relative_ms": f"{self.start_time_ms + (timedelta(seconds=27).total_seconds() * 1000)}",
                "end_frame_relative_ms": f"{self.start_time_ms + (timedelta(minutes=8).total_seconds() * 1000)}",
            },
        )
        Incident.objects.create(
            uuid=uuid4(),
            title="test",
            incident_type=self.def_a_1.incident_type,
            camera=self.camera_a,
            timestamp=self.start_time + timedelta(minutes=16),
            review_status=ReviewStatus.VALID,
            valid_feedback_count=3,
            data={"uuid": "1ee23"},
        )
        Incident.objects.create(
            uuid=uuid4(),
            title="test",
            incident_type=self.def_a_1.incident_type,
            camera=self.camera_a,
            timestamp=self.start_time + timedelta(minutes=30),
            review_status=ReviewStatus.VALID,
            valid_feedback_count=3,
            data={
                "uuid": "1ee23",
                "start_frame_relative_ms": f"{self.start_time_ms + (timedelta(minutes=6).total_seconds() * 1000)}",
                "end_frame_relative_ms": f"{self.start_time_ms}",
            },
        )
        PerceivedActorStateDurationAggregate.objects.create(
            time_bucket_start_timestamp=self.start_time,
            time_bucket_width=TimeBucketWidth.HOUR,
            category=self.def_a_1.perceived_actor_state_duration_category,
            camera_uuid=self.camera_a.uuid,
            duration=timedelta(minutes=150, seconds=4),
        )
        PerceivedActorStateDurationAggregate.objects.create(
            time_bucket_start_timestamp=self.start_time,
            time_bucket_width=TimeBucketWidth.HOUR,
            category=self.def_b_1.perceived_actor_state_duration_category,
            camera_uuid=self.camera_a.uuid,
            duration=timedelta(minutes=80, seconds=56),
        )
        PerceivedActorStateDurationAggregate.objects.create(
            time_bucket_start_timestamp=self.start_time + timedelta(hours=1),
            time_bucket_width=TimeBucketWidth.HOUR,
            category=self.def_a_1.perceived_actor_state_duration_category,
            camera_uuid=self.camera_a.uuid,
            duration=timedelta(minutes=56, seconds=369),
        )
        PerceivedActorStateDurationAggregate.objects.create(
            time_bucket_start_timestamp=self.start_time + timedelta(hours=1),
            time_bucket_width=TimeBucketWidth.HOUR,
            category=self.def_b_1.perceived_actor_state_duration_category,
            camera_uuid=self.camera_a.uuid,
            duration=timedelta(minutes=40, seconds=2),
        )
        Incident.objects.create(
            uuid=uuid4(),
            title="test",
            incident_type=self.def_b_1.incident_type,
            camera=self.camera_a,
            timestamp=self.start_time
            + timedelta(hours=2, minutes=30, seconds=59),
            review_status=ReviewStatus.VALID,
            valid_feedback_count=3,
            data={
                "uuid": "123",
                "start_frame_relative_ms": f"{self.start_time_ms}",
                "end_frame_relative_ms": f"{self.start_time_ms + (timedelta(minutes=2).total_seconds() * 1000)}",
            },
        )
        PerceivedActorStateDurationAggregate.objects.create(
            time_bucket_start_timestamp=self.start_time + timedelta(hours=2),
            time_bucket_width=TimeBucketWidth.HOUR,
            category=self.def_b_1.perceived_actor_state_duration_category,
            camera_uuid=self.camera_a.uuid,
            duration=timedelta(minutes=30, seconds=145),
        )
        Incident.objects.create(
            uuid=uuid4(),
            title="test",
            incident_type=self.def_c_1.incident_type,
            camera=self.camera_b,
            timestamp=self.start_time
            + timedelta(hours=2, minutes=13, seconds=59),
            review_status=ReviewStatus.VALID,
            valid_feedback_count=3,
            data={
                "uuid": "123",
                "start_frame_relative_ms": f"{self.start_time_ms}",
                "end_frame_relative_ms": f"{self.start_time_ms + (timedelta(minutes=14, seconds=31).total_seconds() * 1000)}",
            },
        )
        Incident.objects.create(
            uuid=uuid4(),
            title="test",
            incident_type=self.def_c_1.incident_type,
            camera=self.camera_b,
            timestamp=self.start_time
            + timedelta(hours=2, minutes=59, seconds=59),
            review_status=ReviewStatus.VALID,
            valid_feedback_count=3,
            data={
                "uuid": "123",
                "start_frame_relative_ms": f"{self.start_time_ms}",
                "end_frame_relative_ms": f"{self.start_time_ms + (timedelta(hours=150, minutes=3, seconds=20).total_seconds() * 1000)}",
            },
        )
        PerceivedActorStateDurationAggregate.objects.create(
            time_bucket_start_timestamp=self.start_time + timedelta(hours=2),
            time_bucket_width=TimeBucketWidth.HOUR,
            category=self.def_b_1.perceived_actor_state_duration_category,
            camera_uuid=self.camera_b.uuid,
            duration=timedelta(minutes=187, seconds=23),
        )
        # Unrelated
        PerceivedActorStateDurationAggregate.objects.create(
            time_bucket_start_timestamp=self.start_time - timedelta(seconds=1),
            time_bucket_width=TimeBucketWidth.HOUR,
            category=self.def_a_1.perceived_actor_state_duration_category,
            camera_uuid=self.camera_a.uuid,
            duration=timedelta(minutes=1236, seconds=12),
        )

    def test_hourly_discrete(self):
        # This gets overwritten
        PerceivedEventRateHourly.objects.create(
            camera=self.camera_a,
            time_bucket_start_timestamp=self.start_time + timedelta(hours=1),
            definition=self.def_a_1,
            numerator_value=85,
            denominator_value=timedelta(
                minutes=420, seconds=69
            ).total_seconds(),
        )
        expected = [
            PerceivedEventRateHourly(
                camera=self.camera_a,
                time_bucket_start_timestamp=self.start_time,
                definition=self.def_a_1,
                numerator_value=6,
                denominator_value=timedelta(
                    minutes=150, seconds=4
                ).total_seconds(),
            ),
            PerceivedEventRateHourly(
                camera=self.camera_a,
                time_bucket_start_timestamp=self.start_time,
                definition=self.def_b_1,
                numerator_value=6,
                denominator_value=timedelta(
                    minutes=80, seconds=56
                ).total_seconds(),
            ),
            # This overwrites
            PerceivedEventRateHourly(
                camera=self.camera_a,
                time_bucket_start_timestamp=self.start_time
                + timedelta(hours=1),
                definition=self.def_a_1,
                numerator_value=0,
                denominator_value=timedelta(
                    minutes=56, seconds=369
                ).total_seconds(),
            ),
            PerceivedEventRateHourly(
                camera=self.camera_a,
                time_bucket_start_timestamp=self.start_time
                + timedelta(hours=1),
                definition=self.def_b_1,
                numerator_value=0,
                denominator_value=timedelta(
                    minutes=40, seconds=2
                ).total_seconds(),
            ),
            PerceivedEventRateHourly(
                camera=self.camera_a,
                time_bucket_start_timestamp=self.start_time
                + timedelta(hours=2),
                definition=self.def_b_1,
                numerator_value=1,
                denominator_value=timedelta(
                    minutes=30, seconds=145
                ).total_seconds(),
            ),
            PerceivedEventRateHourly(
                camera=self.camera_b,
                time_bucket_start_timestamp=self.start_time
                + timedelta(hours=2),
                definition=self.def_b_1,
                numerator_value=0,
                denominator_value=timedelta(
                    minutes=187, seconds=23
                ).total_seconds(),
            ),
            PerceivedEventRateHourly(
                camera=self.camera_b,
                time_bucket_start_timestamp=self.start_time
                + timedelta(hours=2),
                definition=self.def_c_1,
                numerator_value=2,
                denominator_value=timedelta(
                    minutes=187, seconds=23
                ).total_seconds(),
            ),
        ]

        PerceivedEventRateCalculationMethodMapping[
            PerceivedEventRateCalculationMethod.HOURLY_DISCRETE
        ](
            calculation_inputs=query_for_camera_calculation_definitions(
                calculation_method=PerceivedEventRateCalculationMethod.HOURLY_DISCRETE,
                camera_ids=[
                    self.camera_a.id,
                    self.camera_b.id,
                    self.camera_c.id,
                ],
            ),
            start_time=self.start_time,
            end_time=self.end_time,
        )

        for expected_obj in expected:
            results_obj = PerceivedEventRateHourly.objects.get(
                camera=expected_obj.camera,
                time_bucket_start_timestamp=expected_obj.time_bucket_start_timestamp,
                definition=expected_obj.definition,
            )
            self.assertEqual(
                results_obj.numerator_value, expected_obj.numerator_value
            )
            self.assertEqual(
                results_obj.denominator_value, expected_obj.denominator_value
            )

        self.assertEqual(
            PerceivedEventRateHourly.objects.all().count(), len(expected)
        )

    def test_hourly_continuous(self):
        # This gets overwritten
        PerceivedEventRateHourly.objects.create(
            camera=self.camera_a,
            time_bucket_start_timestamp=self.start_time + timedelta(hours=1),
            definition=self.def_a_2,
            numerator_value=1050,
            denominator_value=timedelta(
                minutes=420, seconds=69
            ).total_seconds(),
        )

        expected = [
            PerceivedEventRateHourly(
                camera=self.camera_a,
                time_bucket_start_timestamp=self.start_time,
                definition=self.def_a_2,
                numerator_value=timedelta(seconds=633).total_seconds(),
                denominator_value=timedelta(
                    minutes=150, seconds=4
                ).total_seconds(),
            ),
            PerceivedEventRateHourly(
                camera=self.camera_a,
                time_bucket_start_timestamp=self.start_time,
                definition=self.def_b_2,
                numerator_value=timedelta(seconds=633).total_seconds(),
                denominator_value=timedelta(
                    minutes=80, seconds=56
                ).total_seconds(),
            ),
            # This overwrites
            PerceivedEventRateHourly(
                camera=self.camera_a,
                time_bucket_start_timestamp=self.start_time
                + timedelta(hours=1),
                definition=self.def_a_2,
                numerator_value=0,
                denominator_value=timedelta(
                    minutes=56, seconds=369
                ).total_seconds(),
            ),
            PerceivedEventRateHourly(
                camera=self.camera_a,
                time_bucket_start_timestamp=self.start_time
                + timedelta(hours=1),
                definition=self.def_b_2,
                numerator_value=0,
                denominator_value=timedelta(
                    minutes=40, seconds=2
                ).total_seconds(),
            ),
            PerceivedEventRateHourly(
                camera=self.camera_a,
                time_bucket_start_timestamp=self.start_time
                + timedelta(hours=2),
                definition=self.def_b_2,
                numerator_value=timedelta(minutes=2).total_seconds(),
                denominator_value=timedelta(
                    minutes=30, seconds=145
                ).total_seconds(),
            ),
            PerceivedEventRateHourly(
                camera=self.camera_b,
                time_bucket_start_timestamp=self.start_time
                + timedelta(hours=2),
                definition=self.def_b_2,
                numerator_value=0,
                denominator_value=timedelta(
                    minutes=187, seconds=23
                ).total_seconds(),
            ),
            PerceivedEventRateHourly(
                camera=self.camera_b,
                time_bucket_start_timestamp=self.start_time
                + timedelta(hours=2),
                definition=self.def_c_2,
                numerator_value=timedelta(
                    hours=150, minutes=17, seconds=51
                ).total_seconds(),
                denominator_value=timedelta(
                    minutes=187, seconds=23
                ).total_seconds(),
            ),
        ]

        PerceivedEventRateCalculationMethodMapping[
            PerceivedEventRateCalculationMethod.HOURLY_CONTINUOUS
        ](
            calculation_inputs=query_for_camera_calculation_definitions(
                calculation_method=PerceivedEventRateCalculationMethod.HOURLY_CONTINUOUS,
                camera_ids=[
                    self.camera_a.id,
                    self.camera_b.id,
                    self.camera_c.id,
                ],
            ),
            start_time=self.start_time,
            end_time=self.end_time,
        )

        for expected_obj in expected:
            results_obj = PerceivedEventRateHourly.objects.get(
                camera=expected_obj.camera,
                time_bucket_start_timestamp=expected_obj.time_bucket_start_timestamp,
                definition=expected_obj.definition,
            )
            self.assertEqual(
                results_obj.numerator_value, expected_obj.numerator_value
            )
            self.assertEqual(
                results_obj.denominator_value, expected_obj.denominator_value
            )

        self.assertEqual(
            PerceivedEventRateHourly.objects.all().count(), len(expected)
        )


if __name__ == "__main__":
    unittest.main()
