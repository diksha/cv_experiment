// Package clipsynth provides functionality for generating video clips to play in Portal from a fragment archive
package clipsynth

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/rs/zerolog/log"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
	"github.com/voxel-ai/voxel/lib/utils/go/timeutil"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive"
	fragkey "github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive/key"
)

// FragArchiveAPI is the interface for the fragment archive client
//
//go:generate go run github.com/maxbrunsfeld/counterfeiter/v6 . FragArchiveAPI
type FragArchiveAPI interface {
	GetFragment(ctx context.Context, key fragkey.FragmentKey) ([]byte, fragarchive.FragmentMetadata, error)
	GetFragmentKeysInRange(ctx context.Context, query fragarchive.RangeQuery) ([]fragkey.FragmentKey, error)
	GetMetadata(ctx context.Context, key fragkey.FragmentKey) (fragarchive.FragmentMetadata, error)
}

var _ FragArchiveAPI = (*fragarchive.Client)(nil)

// TranscodeFiles transcodes the provided fragment files into a video clip
// fragmentFilePaths must be sorted by start time
func TranscodeFiles(ctx context.Context, fragmentFilePaths []string, startTimestamp, endTimestamp time.Time) ([]byte, error) {
	logger := log.Ctx(ctx)

	outputFile, err := os.CreateTemp("", "clip-*.mp4")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	defer func() {
		err := os.Remove(outputFile.Name())
		if err != nil {
			logger.Warn().
				Err(err).
				Str("outputFile", outputFile.Name()).
				Msg("failed to remove output file")
		}
	}()

	// concatenate fragment file names, delimmited by |
	inputArg := fmt.Sprintf("concat:%s", strings.Join(fragmentFilePaths, "|"))

	startPTS := startTimestamp.UnixMilli()
	endPTS := endTimestamp.UnixMilli()
	filters := []string{
		// select frames between start and end timestamps
		fmt.Sprintf("select='between(pts,%v,%v)'", startPTS, endPTS),
		// Set output timestamps to be relative
		fmt.Sprintf("setpts='PTS-%v'", startPTS),
	}

	args := []string{
		"-nostdin", // disable interaction on stdin
		"-y",       // overwrite output file
		"-i", inputArg,
		"-filter", strings.Join(filters, ","),
		"-copyts",      // copy timestamps from input to output (required for the select filter)
		"-c:v", "h264", // convert to h264
		"-preset", "veryfast",
		outputFile.Name(),
	}

	cmd, err := ffmpeg.New(ctx, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to create ffmpeg command: %w", err)
	}

	// write output to logger
	cmd.Stderr = logger.With().Str("context", "ffmpeg stderr").Logger()

	err = cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("failed to run ffmpeg command: %w", err)
	}

	clipBytes, err := io.ReadAll(outputFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read output file: %w", err)
	}

	return clipBytes, nil
}

// GenerateClip generates a video clip from the fragment archive
func GenerateClip(ctx context.Context, archive FragArchiveAPI, cameraUUID string, startTimestamp, endTimestamp time.Time) ([]byte, error) {
	logger := log.Ctx(ctx)

	// Round the start and end timestamps without truncating the clip
	startTimestamp = startTimestamp.Truncate(time.Second)
	endTimestamp = timeutil.RoundUp(endTimestamp, time.Second)

	// Get the fragment keys for the requested time range
	fragmentKeys, err := archive.GetFragmentKeysInRange(ctx, fragarchive.RangeQuery{
		CameraUUID: cameraUUID,
		StartTime:  startTimestamp,
		EndTime:    endTimestamp,
	})
	if err != nil {
		logger.Debug().
			Err(err).
			Str("cameraUUID", cameraUUID).
			Time("startTimestamp", startTimestamp).
			Time("endTimestamp", endTimestamp).
			Msg("failed to get fragment keys")

		return nil, fmt.Errorf("failed to get fragment keys: %w", err)
	}

	// Get the metadata for the fragments
	metadatas, err := getFragmentsMetadata(ctx, archive, fragmentKeys)
	if err != nil {
		return nil, fmt.Errorf("failed to get metadata for fragments: %w", err)
	}

	// Validate the metadata for the fragments
	if err = sanityCheckFragmentsMetadata(ctx, metadatas); err != nil {
		return nil, err
	}

	// Download files for the fragments
	fragmentFilePaths := []string{}
	fragmentDir, err := os.MkdirTemp("", "fragments-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer func() {
		if err := os.RemoveAll(fragmentDir); err != nil {
			logger.Warn().
				Err(err).
				Str("fragmentDir", fragmentDir).
				Msg("failed to remove fragment dir")
		}
	}()

	for _, fragmentKey := range fragmentKeys {
		ctxWithFragkey := logger.With().Stringer("fragmentKey", &fragmentKey).Logger().WithContext(ctx)

		// Download fragment from archive
		fragmentData, _, err := archive.GetFragment(ctxWithFragkey, fragmentKey)
		if err != nil {
			logger.Debug().
				Err(err).
				Msg("failed to get fragment")

			return nil, fmt.Errorf("failed to get fragment: %w", err)
		}

		// Write fragment to temp file
		file, err := writeFragmentToNewFile(ctxWithFragkey, fragmentData, fragmentDir)
		if err != nil {
			return nil, fmt.Errorf("failed to write fragment to temp file: %w", err)
		}

		fragmentFilePaths = append(fragmentFilePaths, file)
	}

	return TranscodeFiles(ctx, fragmentFilePaths, startTimestamp, endTimestamp)
}

func writeFragmentToNewFile(ctx context.Context, fragmentData []byte, tempDir string) (string, error) {
	logger := log.Ctx(ctx)

	fragmentFile, err := os.CreateTemp(tempDir, "fragment-*.mkv")
	if err != nil {
		logger.Debug().
			Err(err).
			Msg("failed to create temp file")

		return "", fmt.Errorf("failed to create temp file: %w", err)
	}

	defer func() {
		if err := fragmentFile.Close(); err != nil {
			logger.Warn().
				Err(err).
				Msg("failed to close temp file")
		}
	}()

	err = os.WriteFile(fragmentFile.Name(), fragmentData, 0600)
	if err != nil {
		logger.Debug().
			Err(err).
			Msg("failed to write fragment to temp file")

		return "", fmt.Errorf("failed to write fragment to temp file: %w", err)
	}

	return fragmentFile.Name(), nil
}

func getFragmentsMetadata(ctx context.Context, archive FragArchiveAPI, keys []fragkey.FragmentKey) ([]fragarchive.FragmentMetadata, error) {
	logger := log.Ctx(ctx)

	metadatas := []fragarchive.FragmentMetadata{}
	for _, key := range keys {
		metadata, err := archive.GetMetadata(ctx, key)
		if err != nil {
			logger.Debug().
				Err(err).
				Stringer("fragmentKey", &key).
				Msg("failed to get fragment metadata")

			return nil, fmt.Errorf("failed to get fragment metadata: %w", err)
		}

		metadatas = append(metadatas, metadata)
	}

	return metadatas, nil
}

func sanityCheckFragmentsMetadata(ctx context.Context, metadatas []fragarchive.FragmentMetadata) error {
	timeSeries := timeutil.TimeRangeSeries{}
	for _, metadata := range metadatas {
		timeSeries = append(timeSeries, timeutil.NewTimeRangeFromDuration(metadata.ProducerTimestamp, metadata.Duration))
	}

	// Sanity check
	if !timeSeries.IsSortedByStart() {
		return fmt.Errorf("fragments are not sorted by start time")
	}

	return nil
}
