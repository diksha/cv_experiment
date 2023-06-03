package fragarchive

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/rs/zerolog/log"

	fragkey "github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive/key"
)

const (
	metadataDurationKey          = "x-amz-meta-duration-ms"
	metadataFragmentNumberKey    = "x-amz-meta-fragment-number"
	metadataProducerTimestampKey = "x-amz-meta-producer-timestamp-ms"
	metadataServerTimestampKey   = "x-amz-meta-server-timestamp-ms"
	metadataCameraUUIDKey        = "x-amz-meta-camera-uuid"

	// metadata validation constants
	maximumFragmentDuration      = 10 * time.Second
	maximumProducerServerLatency = 1 * time.Hour // arbitrary limit

)

// FragmentMetadata is the metadata associated with a fragment
type FragmentMetadata struct {
	Duration          time.Duration
	FragmentNumber    string
	ProducerTimestamp time.Time
	ServerTimestamp   time.Time
	CameraUUID        string
}

func (metadata FragmentMetadata) validate(ctx context.Context) error {
	logger := log.Ctx(ctx)

	if metadata.Duration <= 0 {
		return fmt.Errorf("duration must be positive")
	} else if metadata.Duration > maximumFragmentDuration {
		return fmt.Errorf("duration must not be greater than %v", maximumFragmentDuration)
	}

	if err := fragkey.ValidateTimestamp(metadata.ProducerTimestamp); err != nil {
		return fmt.Errorf("invalid producer timestamp: %w", err)
	}

	if err := fragkey.ValidateTimestamp(metadata.ServerTimestamp); err != nil {
		logger.Warn().
			Err(err).
			Time("server timestamp", metadata.ServerTimestamp).
			Msg("invalid server timestamp")
	}

	if metadata.CameraUUID == "" {
		return fmt.Errorf("camera UUID must be non-empty")
	}

	if metadata.ServerTimestamp.Before(metadata.ProducerTimestamp) {
		logger.Warn().
			Time("producer timestamp", metadata.ProducerTimestamp).
			Time("server timestamp", metadata.ServerTimestamp).
			Msg("server timestamp is before producer timestamp")
	}

	timeDelta := metadata.ServerTimestamp.Sub(metadata.ProducerTimestamp)
	if timeDelta > maximumProducerServerLatency {
		logger.Warn().
			Time("producer timestamp", metadata.ProducerTimestamp).
			Time("server timestamp", metadata.ServerTimestamp).
			Dur("latency", timeDelta).
			Msg("large latency between producer and server timestamps")
	}

	return nil
}

func (metadata *FragmentMetadata) toMap() map[string]string {
	return map[string]string{
		metadataDurationKey:          strconv.FormatInt(metadata.Duration.Milliseconds(), 10),
		metadataFragmentNumberKey:    metadata.FragmentNumber,
		metadataProducerTimestampKey: strconv.FormatInt(metadata.ProducerTimestamp.UnixMilli(), 10),
		metadataServerTimestampKey:   strconv.FormatInt(metadata.ServerTimestamp.UnixMilli(), 10),
		metadataCameraUUIDKey:        metadata.CameraUUID,
	}
}

// ParseFragmentMetadata parses fragment metadata from a map of strings.
// Will return an error if any of the metadata cannot be parsed to the correct type.
// Will not return an error on missing metadata.
// Will not validate any of the successfully parsed metadata.
func ParseFragmentMetadata(metadata map[string]string) (FragmentMetadata, error) {
	result := FragmentMetadata{}

	if durationStr, ok := metadata[metadataDurationKey]; ok {
		duration, err := strconv.ParseInt(durationStr, 10, 64)
		if err != nil {
			return FragmentMetadata{}, fmt.Errorf("failed to parse duration: %w", err)
		}
		result.Duration = time.Duration(duration) * time.Millisecond
	}

	if fragmentNumber, ok := metadata[metadataFragmentNumberKey]; ok {
		result.FragmentNumber = fragmentNumber
	}

	if producerTimestampStr, ok := metadata[metadataProducerTimestampKey]; ok {
		producerTimestampMs, err := strconv.ParseInt(producerTimestampStr, 10, 64)
		if err != nil {
			return FragmentMetadata{}, fmt.Errorf("failed to parse producer timestamp: %w", err)
		}
		result.ProducerTimestamp = time.UnixMilli(producerTimestampMs)
	}

	if serverTimestampStr, ok := metadata[metadataServerTimestampKey]; ok {
		serverTimestampMs, err := strconv.ParseInt(serverTimestampStr, 10, 64)
		if err != nil {
			return FragmentMetadata{}, fmt.Errorf("failed to parse server timestamp: %w", err)
		}
		result.ServerTimestamp = time.UnixMilli(serverTimestampMs)
	}

	if cameraUUID, ok := metadata[metadataCameraUUIDKey]; ok {
		result.CameraUUID = cameraUUID
	}

	return result, nil
}
