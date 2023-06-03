// Package incident provides functionality for working with incidents and deserializing
package incident

import (
	"encoding/json"
	"fmt"
	"time"
)

// Incident is the struct representation of a incident sent to be published
type Incident struct {
	CameraUUID           string `json:"camera_uuid"`
	StreamARN            string `json:"kvs_stream_arn"`
	StartFrameRelativeMs int    `json:"start_frame_relative_ms"`
	EndFrameRelativeMs   int    `json:"end_frame_relative_ms"`
	PreStartBufferMs     int    `json:"pre_start_buffer_ms"`
	PostEndBufferMs      int    `json:"post_end_buffer_ms"`
}

// Unmarshal creates an Incident from a JSON string
func Unmarshal(jsonEncoded string) (*Incident, error) {
	var incident Incident
	if err := json.Unmarshal([]byte(jsonEncoded), &incident); err != nil {
		return nil, fmt.Errorf("failed to unmarshall JSON-encoded incident: %w", err)
	}

	return &incident, nil
}

// GetClipStartTime returns the buffered start time of the incident
func (incident *Incident) GetClipStartTime() time.Time {
	return time.UnixMilli(int64(incident.StartFrameRelativeMs - incident.PreStartBufferMs))
}

// GetClipEndTime returns the buffered end time of the incident
func (incident *Incident) GetClipEndTime() time.Time {
	return time.UnixMilli(int64(incident.EndFrameRelativeMs + incident.PostEndBufferMs))
}
