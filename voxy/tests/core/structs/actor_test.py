#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import unittest
import uuid

from core.structs.actor import (
    Actor,
    ActorCategory,
    ActorFactory,
    HeadCoveringType,
    MotionDetectionZoneState,
    get_actor_id_from_actor_category_and_track_id,
    get_track_uuid,
)
from core.structs.ergonomics import Activity, ActivityType, PostureType

# trunk-ignore-begin(pylint/E0611)
from protos.perception.structs.v1.actor_pb2 import (
    ErgonomicSeverityMetrics as ErgonomicSeverityMetricsPb,
)
from protos.perception.structs.v1.actor_pb2 import Limb as LimbPb
from protos.perception.structs.v1.actor_pb2 import LimbType as LimbTypePb
from protos.perception.structs.v1.actor_pb2 import (
    PoseKeypointType as PoseKeypointTypePb,
)
from protos.perception.structs.v1.actor_pb2 import RebaScores as RebaScoresPb
from protos.perception.structs.v1.actor_pb2 import Skeleton as SkeletonPb

# trunk-ignore-end(pylint/E0611)


class ActorTest(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up test...")
        self.actor = Actor()

    def testActor(self) -> None:
        """
        Tests actor serialization
        """
        # test empty actor
        actor_dict = self.actor.to_dict()
        self.assertSetEqual(set(map(type, actor_dict.values())), {type(None)})

        # check conversion back
        actor = self.actor.from_dict(data=actor_dict)
        self.assertEqual(actor, self.actor)

        # test actor with activity
        self.actor.activity = {
            "LIFTING": Activity(
                ActivityType(ActivityType.LIFTING),
                PostureType(PostureType.GOOD.value),
            ),
            "REACHING": Activity(
                ActivityType(ActivityType.REACHING),
                PostureType(PostureType.BAD.value),
            ),
        }

        actor_dict = self.actor.to_dict()

        self.assertEqual(
            self.actor.activity["LIFTING"],
            Activity(
                ActivityType(ActivityType.LIFTING),
                PostureType(PostureType.GOOD.value),
            ),
        )

        self.assertEqual(
            self.actor.activity["REACHING"],
            Activity(
                ActivityType(ActivityType.REACHING),
                PostureType(PostureType.BAD.value),
            ),
        )

        actor = Actor.from_dict(data=actor_dict)
        self.assertEqual(actor, self.actor)

        # test is_wearing_hard_hat migration to head_covering_type
        self.actor.is_wearing_hard_hat = False
        actor_dict = self.actor.to_dict()
        self.actor.head_covering_type = HeadCoveringType.LEGACY_BARE_HEAD
        actor = Actor.from_dict(data=actor_dict)
        self.assertEqual(actor, self.actor)
        self.assertEqual(
            actor.head_covering_type, HeadCoveringType.LEGACY_BARE_HEAD
        )

        self.actor.is_wearing_hard_hat = True
        self.actor.head_covering_type = None
        actor_dict = self.actor.to_dict()
        self.actor.head_covering_type = HeadCoveringType.HARD_HAT
        actor = Actor.from_dict(data=actor_dict)
        self.assertEqual(actor, self.actor)
        self.assertEqual(actor.head_covering_type, HeadCoveringType.HARD_HAT)

        self.actor.is_wearing_hard_hat = False
        self.actor.head_covering_type = HeadCoveringType.BARE_HEAD
        actor_dict = self.actor.to_dict()
        actor = Actor.from_dict(data=actor_dict)
        self.assertEqual(actor, self.actor)
        self.assertEqual(actor.head_covering_type, HeadCoveringType.BARE_HEAD)

        # test skeleton
        neck = LimbPb(
            joint_start=PoseKeypointTypePb.POSE_KEYPOINT_TYPE_NOSE,
            joint_end=PoseKeypointTypePb.POSE_KEYPOINT_TYPE_NECK,
            confidence_probability=0.95,
            limb_type=LimbTypePb.LIMB_TYPE_NECK,
        )
        right_shoulder = LimbPb(
            joint_start=PoseKeypointTypePb.POSE_KEYPOINT_TYPE_NECK,
            joint_end=PoseKeypointTypePb.POSE_KEYPOINT_TYPE_RIGHT_SHOULDER,
            confidence_probability=0.95,
            limb_type=LimbTypePb.LIMB_TYPE_TRUNK,
        )
        skeleton = SkeletonPb(
            limbs=[neck, right_shoulder],
        )
        self.actor.skeleton = skeleton
        actor_dict = self.actor.to_dict()
        actor = Actor.from_dict(data=actor_dict)
        self.assertEqual(actor, self.actor)

        # test ergonomic_severity_metrics
        severity_metrics = ErgonomicSeverityMetricsPb(
            reba_scores=RebaScoresPb(
                version="0.1",
                neck=1,
                upper_arms=1,
                lower_arms=1,
                trunk=1,
                legs=1,
                table_a=1,
                table_b=1,
                table_c=1,
            )
        )
        self.actor.ergonomic_severity_metrics = severity_metrics
        actor_dict = self.actor.to_dict()
        actor = Actor.from_dict(data=actor_dict)
        self.assertEqual(actor, self.actor)

    # trunk-ignore-all(pylint/C0103)
    def testSerialization(self):
        """
        Tests to dict, to proto and from proto methods
        """

        # validate that the to_dict method works as expected
        test_person_actor_without_pose = Actor(
            uuid=str(uuid.uuid4()),
            category=ActorCategory.PERSON,
            track_id=5,
            track_uuid=str(uuid.uuid4()),
            pose=None,  # TODO: check pose
            confidence=1.0,
            occluded_degree=None,  # TODO: check occluded_degree
            distance_to_camera_m=1.0,
            is_wearing_safety_vest=True,
            is_wearing_hard_hat=True,
            is_carrying_object=False,
        )
        # check conversion back
        actor = Actor.from_dict(data=test_person_actor_without_pose.to_dict())
        self.assertEqual(
            actor.track_uuid, test_person_actor_without_pose.track_uuid
        )
        self.assertEqual(
            actor.to_dict(), test_person_actor_without_pose.to_dict()
        )
        self.maxDiff = None

        # now convert to a dict:
        # actor -(to_proto())-> proto -(from_proto())->
        # actor -(to_dict())-> dict
        # and compare the dicts
        actor_proto = test_person_actor_without_pose.to_proto()
        # add back when from_proto method is finished
        from_proto_actor = Actor.from_proto(actor_proto)
        self.assertEqual(
            from_proto_actor.to_dict(),
            test_person_actor_without_pose.to_dict(),
        )

        # validate with skeleton
        neck = LimbPb(
            joint_start=PoseKeypointTypePb.POSE_KEYPOINT_TYPE_NOSE,
            joint_end=PoseKeypointTypePb.POSE_KEYPOINT_TYPE_NECK,
            confidence_probability=0.95,
            limb_type=LimbTypePb.LIMB_TYPE_NECK,
        )
        right_shoulder = LimbPb(
            joint_start=PoseKeypointTypePb.POSE_KEYPOINT_TYPE_NECK,
            joint_end=PoseKeypointTypePb.POSE_KEYPOINT_TYPE_RIGHT_SHOULDER,
            confidence_probability=0.95,
            limb_type=LimbTypePb.LIMB_TYPE_TRUNK,
        )
        skeleton = SkeletonPb(
            limbs=[neck, right_shoulder],
        )
        test_person_actor_without_pose.skeleton = skeleton
        actor_proto = test_person_actor_without_pose.to_proto()
        from_proto_actor = Actor.from_proto(actor_proto)
        self.assertEqual(
            from_proto_actor.to_dict(),
            test_person_actor_without_pose.to_dict(),
        )

        # validate with ergo severity metrics
        severity_metrics = ErgonomicSeverityMetricsPb(
            reba_scores=RebaScoresPb(
                version="0.1",
                neck=1,
                upper_arms=1,
                lower_arms=1,
                trunk=1,
                legs=1,
                table_a=1,
                table_b=1,
                table_c=1,
            )
        )
        test_person_actor_without_pose.ergonomic_severity_metrics = (
            severity_metrics
        )
        actor_proto = test_person_actor_without_pose.to_proto()
        from_proto_actor = Actor.from_proto(actor_proto)
        self.assertEqual(
            from_proto_actor.to_dict(),
            test_person_actor_without_pose.to_dict(),
        )

    def testProductionLineActorProtos(self) -> None:
        """
        Tests end to end production line serialization/deserialization
        """
        for score_value in [None, 0.0, 1.0]:
            for zone_type in [
                MotionDetectionZoneState.FROZEN,
                MotionDetectionZoneState.MOTION,
                MotionDetectionZoneState.UNKNOWN,
            ]:
                test_zone_actor = Actor(
                    uuid=str(uuid.uuid4()),
                    category=ActorCategory.MOTION_DETECTION_ZONE,
                    track_id=5,
                    track_uuid=str(uuid.uuid4()),
                    motion_detection_zone_state=zone_type,
                    motion_detection_score_std=score_value,
                )
                actor_proto = test_zone_actor.to_proto()
                # add back when from_proto method is finished
                from_proto_actor = Actor.from_proto(actor_proto)
                self.assertEqual(
                    from_proto_actor.to_dict(),
                    test_zone_actor.to_dict(),
                )

    def testActorFactory(self) -> None:
        """
        Tests creating actors using the factory
        """
        actor = ActorFactory.from_detection(
            "camera_uuid",
            1,
            [1, 2, 3, 4],
            ActorCategory.DOOR,
            0.5,
        )
        self.assertEqual(
            actor.track_uuid, "454238a5-d3d6-4157-a473-881cc8e47d8e"
        )

    def testTrackUuid(self) -> None:
        """
        Tests creating track uuids
        """
        track_uuid = get_track_uuid(
            "americold/modesto/0001/cha", "1", ActorCategory.DOOR
        )
        self.assertTrue("3041c916-9b45-47f0-b264-5fee089a3fd0" == track_uuid)

    def testGetActorIdFromActorCategoryAndTrackId(self) -> None:
        """
        Tests creating actor ids
        """

        # Test door
        actor_id = get_actor_id_from_actor_category_and_track_id(
            1,
            ActorCategory.DOOR,
        )
        self.assertTrue(3001 == actor_id)

        # test aisle
        actor_id = get_actor_id_from_actor_category_and_track_id(
            1, ActorCategory.AISLE_END
        )
        self.assertTrue(9001 == actor_id)

        # test intersection
        actor_id = get_actor_id_from_actor_category_and_track_id(
            1, ActorCategory.INTERSECTION
        )
        self.assertTrue(8001 == actor_id)

        # test driving area
        actor_id = get_actor_id_from_actor_category_and_track_id(
            1, ActorCategory.DRIVING_AREA
        )
        self.assertTrue(13001 == actor_id)

        # test no ped zone
        actor_id = get_actor_id_from_actor_category_and_track_id(
            1, ActorCategory.NO_PED_ZONE
        )
        self.assertTrue(12001 == actor_id)


if __name__ == "__main__":
    unittest.main()
