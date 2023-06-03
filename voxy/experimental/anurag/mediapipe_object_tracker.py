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
"""Provides an implementation of object tracking using TF and TF-TRT."""
import numpy as np
from google.protobuf import text_format
from mediapipe.calculators.video.box_tracker_calculator_pb2 import (
    BoxTrackerCalculatorOptions,
)
from mediapipe.calculators.video.flow_packager_calculator_pb2 import (
    FlowPackagerCalculatorOptions,
)
from mediapipe.calculators.video.motion_analysis_calculator_pb2 import (
    MotionAnalysisCalculatorOptions,
)
from mediapipe.calculators.video.tracked_detection_manager_calculator_pb2 import (
    TrackedDetectionManagerCalculatorOptions,
)
from mediapipe.framework import calculator_pb2
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats.detection_pb2 import Detection
from mediapipe.framework.formats.detection_pb2 import DetectionList
from mediapipe.framework.formats.location_data_pb2 import LocationData
from mediapipe.framework.stream_handler.sync_set_input_stream_handler_pb2 import (
    SyncSetInputStreamHandlerOptions,
)
from mediapipe.python import solution_base

from .base_object_detection import BaseObjectDetectionInference
from .types import NormalizedBoundingBox
from .types import ObjectTrackingAnnotation

mediapipe_graph = """
input_stream: "input_frame"
input_stream: "input_detections"
output_stream: "output_tracked"

# Assigns an unique id for each new detection.
node {
  calculator: "DetectionUniqueIdCalculator"
  input_stream: "DETECTION_LIST:input_detections"
  output_stream: "DETECTION_LIST:detections_with_id"
}

# Converts detections to TimedBox protos which are used as initial location
# for tracking.
node {
  calculator: "DetectionsToTimedBoxListCalculator"
  input_stream: "DETECTION_LIST:detections_with_id"
  output_stream: "BOXES:start_pos"
}


node: {
  calculator: "ImageTransformationCalculator"
  input_stream: "IMAGE:input_frame"
  output_stream: "IMAGE:downscaled_input_video"
  node_options: {
    [type.googleapis.com/mediapipe.ImageTransformationCalculatorOptions] {
      output_width: 640
      output_height: 360
    }
  }
}

# Performs motion analysis on an incoming video stream.
node: {
  calculator: "MotionAnalysisCalculator"
  input_stream: "VIDEO:downscaled_input_video"
  output_stream: "CAMERA:camera_motion"
  output_stream: "FLOW:region_flow"

  node_options: {
    [type.googleapis.com/mediapipe.MotionAnalysisCalculatorOptions]: {
      analysis_options {
        analysis_policy: ANALYSIS_POLICY_CAMERA_MOBILE
        flow_options {
          fast_estimation_min_block_size: 100
          top_inlier_sets: 1
          frac_inlier_error_threshold: 3e-3
          downsample_mode: DOWNSAMPLE_TO_INPUT_SIZE
          # downsample_factor: 2.0
          verification_distance: 5.0
          verify_long_feature_acceleration: true
          verify_long_feature_trigger_ratio: 0.1
          tracking_options {
            max_features: 2000
            min_feature_distance: 4
            reuse_features_max_frame_distance: 3
            reuse_features_min_survived_frac: 0.9
            adaptive_extraction_levels: 2
            min_eig_val_settings {
              adaptive_lowest_quality_level: 2e-4
            }
            klt_tracker_implementation: KLT_OPENCV
          }
        }
        motion_options {
          label_empty_frames_as_valid: false
        }
      }
    }
  }
}

# Reads optical flow fields defined in
# mediapipe/framework/formats/motion/optical_flow_field.h,
# returns a VideoFrame with 2 channels (v_x and v_y), each channel is quantized
# to 0-255.
node: {
  calculator: "FlowPackagerCalculator"
  input_stream: "FLOW:region_flow"
  input_stream: "CAMERA:camera_motion"
  output_stream: "TRACKING:tracking_data"

  node_options: {
    [type.googleapis.com/mediapipe.FlowPackagerCalculatorOptions]: {
      flow_packager_options: {
        binary_tracking_data_support: false
      }
    }
  }
}

# Tracks box positions over time.
node: {
  calculator: "BoxTrackerCalculator"
  input_stream: "TRACKING:tracking_data"
  input_stream: "TRACK_TIME:input_frame"
  input_stream: "START_POS:start_pos"
  input_stream: "CANCEL_OBJECT_ID:cancel_object_id"
  input_stream_info: {
    tag_index: "CANCEL_OBJECT_ID"
    back_edge: true
  }
  output_stream: "BOXES:boxes"

  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler"
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "TRACKING"
          tag_index: "TRACK_TIME"
          tag_index: "START_POS"
        }
        sync_set {
          tag_index: "CANCEL_OBJECT_ID"
        }
      }
    }
  }

  node_options: {
    [type.googleapis.com/mediapipe.BoxTrackerCalculatorOptions]: {
      tracker_options: {
        track_step_options {
          track_object_and_camera: true
          tracking_degrees: TRACKING_DEGREE_TRANSLATION
          inlier_spring_force: 0.0
          static_motion_temporal_ratio: 3e-2
        }
      }
      visualize_tracking_data: false
      streaming_track_data_cache_size: 100
    }
  }
}


# Managers new detected objects and objects that are being tracked.
# It associates the duplicated detections and updates the locations of
# detections from tracking.
node: {
  calculator: "TrackedDetectionManagerCalculator"
  input_stream: "DETECTION_LIST:detections_with_id"
  input_stream: "TRACKING_BOXES:boxes"
  output_stream: "DETECTIONS:output_tracked"
  output_stream: "CANCEL_OBJECT_ID:cancel_object_id"

  options: {
    [mediapipe.TrackedDetectionManagerCalculatorOptions.ext]: {
      tracked_detection_manager_options {
        is_same_detection_min_overlap_ratio: 0.15
      }
    }
  }

  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler"
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "TRACKING_BOXES"
          tag_index: "DETECTION_LIST"
        }
      }
    }
  }
}
"""

config_proto = text_format.Parse(
    mediapipe_graph, calculator_pb2.CalculatorGraphConfig()
)
mediapipe_tracker = solution_base.SolutionBase(graph_config=config_proto)


class MediaPipeObjectTracker(BaseObjectDetectionInference):
    """MediaPipe-based tracking."""

    def __init__(self, object_detection_engine, config):
        del config  # Not used, yet.
        self._object_detection_engine = object_detection_engine

    def input_size(self):
        return self._object_detection_engine.input_size()

    def run(self, timestamp, frame, annotations):
        np_frame = np.array(frame)

        detection_annotations = []
        if self._object_detection_engine.run(
            timestamp, np_frame, detection_annotations
        ):

            detection_list = DetectionList()
            for idx, annotation in enumerate(detection_annotations):
                detection = Detection()
                detection.timestamp_usec = timestamp
                detection.label.extend([annotation.class_name])
                detection.score.extend([annotation.confidence_score])
                detection.detection_id = idx
                detection.location_data.relative_bounding_box.xmin = (
                    annotation.bbox.left
                )
                detection.location_data.relative_bounding_box.ymin = annotation.bbox.top
                detection.location_data.relative_bounding_box.width = (
                    annotation.bbox.right - annotation.bbox.left
                )
                detection.location_data.relative_bounding_box.height = (
                    annotation.bbox.bottom - annotation.bbox.top
                )
                detection_list.detection.append(detection)

            # Inputs annotations into mediapipe tracker.
            tracked_annotations = mediapipe_tracker.process(
                {"input_detections": detection_list, "input_frame": np_frame},
                timestamp=timestamp,
            )

            # Converts back to AutoML Video Edge detection structs.
            for tracked_annotation in tracked_annotations.output_tracked:
                output_annotation = ObjectTrackingAnnotation(
                    timestamp=timestamp,
                    track_id=tracked_annotation.detection_id,
                    class_id=1 if tracked_annotation.label_id else -1,
                    class_name=tracked_annotation.label,
                    confidence_score=float(tracked_annotation.score[0]),
                    bbox=NormalizedBoundingBox(
                        left=tracked_annotation.location_data.relative_bounding_box.xmin,
                        top=tracked_annotation.location_data.relative_bounding_box.ymin,
                        right=tracked_annotation.location_data.relative_bounding_box.xmin
                        + tracked_annotation.location_data.relative_bounding_box.width,
                        bottom=tracked_annotation.location_data.relative_bounding_box.ymin
                        + tracked_annotation.location_data.relative_bounding_box.height,
                    ),
                )
                annotations.append(output_annotation)
            return True
        else:
            return False
