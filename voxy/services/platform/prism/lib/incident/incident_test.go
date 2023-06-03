package incident_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/services/platform/prism/lib/incident"
)

const sampleIncidentJSON string = "{\"uuid\": \"1da3457c-e063-4995-a218-24dcf0ad28b4\", \"camera_uuid\": \"wesco/reno/0002/cha\", \"kvs_stream_arn\": \"arn:aws:kinesisvideo:us-west-2:360054435465:stream/uscold-laredo-0001/1651957466439\", \"camera_config_version\": 1, \"organization_key\": \"VOXEL_SANDBOX\", \"title\": \"Safety Vest\", \"incident_type_id\": \"SAFETY_VEST\", \"incident_version\": \"1.0\", \"priority\": \"low\", \"actor_ids\": [\"155\"], \"video_thumbnail_gcs_path\": null, \"video_thumbnail_s3_path\": null, \"video_gcs_path\": null, \"video_s3_path\": null, \"original_video_gcs_path\": null, \"original_video_s3_path\": null, \"annotations_gcs_path\": null, \"annotations_s3_path\": null, \"start_frame_relative_ms\": 1677798504647, \"end_frame_relative_ms\": 1677798509839, \"incident_group_start_time_ms\": null, \"pre_start_buffer_ms\": 11000, \"post_end_buffer_ms\": 6000, \"docker_image_tag\": null, \"cooldown_tag\": false, \"track_uuid\": \"99a8d57d-ec21-4771-853b-2a114d4abf09\", \"sequence_id\": null, \"run_uuid\": \":f75dd44e-b666-4f62-b817-5d56eba777a1\"}"

var sampleIncident incident.Incident = incident.Incident{
	CameraUUID:           "wesco/reno/0002/cha",
	StreamARN:            "arn:aws:kinesisvideo:us-west-2:360054435465:stream/uscold-laredo-0001/1651957466439",
	StartFrameRelativeMs: 1677798504647,
	EndFrameRelativeMs:   1677798509839,
	PreStartBufferMs:     11000,
	PostEndBufferMs:      6000,
}

func TestUnmarshal(t *testing.T) {
	parsedIncident, err := incident.Unmarshal(sampleIncidentJSON)
	require.NoError(t, err)
	assert.Equal(t, sampleIncident, *parsedIncident)
}

func TestGetClipStartTime(t *testing.T) {
	assert.Equal(t, sampleIncident.GetClipStartTime(), time.UnixMilli(1677798504647-11000))
}

func TestGetClipEndTime(t *testing.T) {
	assert.EqualValues(t, sampleIncident.GetClipEndTime(), time.UnixMilli(1677798509839+6000))
}
