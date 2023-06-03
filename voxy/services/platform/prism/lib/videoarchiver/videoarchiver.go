// Package videoarchiver provides functionality for downloading video fragments from Kineses Video to a fragment archive
package videoarchiver

import (
	"context"
	"errors"
	"fmt"

	// trunk-ignore(semgrep/go.lang.security.audit.crypto.math_random.math-random-used)
	"math/rand"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideoarchivedmedia/types"
	"github.com/rs/zerolog/log"

	"github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive"
	fragkey "github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive/key"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/internal/fragutils"
)

const maximumFragmentDuration = 10 * time.Second

// Client is the interface for serializing fragments from Kinesis to a fragment archive
type Client struct {
	cameraUUID string
	kvsClient  *kvsClient
	archive    *fragarchive.Client
}

// New creates a new fragment archiver client
func New(ctx context.Context, config Config) (*Client, error) {
	streamID, err := config.getStreamIdentifier()
	if err != nil {
		return nil, err
	}
	kvsClient, err := newKVSClient(ctx, streamID, config.KinesisVideoClient, config.KinesisVideoArchivedMediaClient)
	if err != nil {
		return nil, err
	}

	return &Client{
		cameraUUID: config.CameraUUID,
		kvsClient:  kvsClient,
		archive:    config.FragmentArchiveClient,
	}, nil
}

// ArchiveTimeRange archives all fragments from the Kinesis Video stream in the provided time range
func (client *Client) ArchiveTimeRange(ctx context.Context, startTime, endTime time.Time) error {
	logger := log.Ctx(ctx)
	fragmentSeries, err := client.getFragmentSeriesForClip(ctx, startTime, endTime)
	if err != nil {
		return err
	}

	// Total failure case
	if fragmentSeries.IsEmpty() {
		return errors.New("unable to fetch any fragments from Kinesis Video in the provided time range")
	}

	numFailedChecks := 0
	// Failure cases where we can continue with best effort
	if gaps := fragmentSeries.CheckContinuity(); gaps != nil {
		numFailedChecks++
		for _, gap := range gaps {
			logger.Warn().
				Time("gap start", gap.Start).
				Time("gap end", gap.End).
				Msg("Coverage gap detected in fragments fetched from KVS")
		}
	}
	if !fragmentSeries.CheckIncludesStart(startTime) {
		numFailedChecks++
		logger.Error().
			Time("query start time", startTime).
			Msg("fragments fetched from KVS do not have coverage of start time")
	}
	if !fragmentSeries.CheckIncludesEnd(endTime) {
		numFailedChecks++
		logger.Error().
			Time("query end time", endTime).
			Msg("fragments fetched from KVS do not have coverage of end time")
	}

	if numFailedChecks != 0 {
		logger.Error().
			Stringer("Fragment Series", &fragmentSeries).
			Int("num failed checks", numFailedChecks).
			Msg("fragment series failed continuity checks")
	}

	// shuffle fragments
	shuffledFrags := make([]types.Fragment, len(fragmentSeries.Fragments))
	copy(shuffledFrags, fragmentSeries.Fragments)
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(shuffledFrags), func(i, j int) {
		shuffledFrags[i], shuffledFrags[j] = shuffledFrags[j], shuffledFrags[i]
	})

	// Archive Fragments
	failureCount := 0
	for _, fragment := range shuffledFrags {
		key, err := fragkey.NewFromFragment(client.cameraUUID, fragment, fragkey.FormatMKV)
		if err != nil {
			logger.Error().Err(err).Msg("failed to archive fragment")
			failureCount++
			continue
		}

		alreadyArchived, err := client.archive.FragmentExists(ctx, key)
		if err != nil {
			logger.Error().
				Err(err).
				Stringer("fragment key", &key).
				Msg("failed to check if object exists in archive")
			logger.Info().
				Msg("proceeding with upload attempt...")

			// redundant in current archive implementation, but may not be in future
			alreadyArchived = false
		}

		if !alreadyArchived {
			err = client.putFragment(ctx, key, fragment)
			if err != nil {
				// print instead of throwing for best-effort service
				logger.Error().
					Err(err).
					Stringer("fragment key", &key).
					Msg("failed to put fragment to archive")
				failureCount++
			}
		}
	}

	if failureCount != 0 {
		return fmt.Errorf("failed to put %v fragments to archive", failureCount)
	}

	return nil
}

func (client *Client) getFragmentSeriesForClip(ctx context.Context, startTime, endTime time.Time) (fragutils.FragmentSeries, error) {
	fragments, err := client.kvsClient.ListFragments(
		ctx,
		// add buffering since will only return
		startTime.Add(-maximumFragmentDuration),

		// no buffering since this works on
		endTime,
	)
	if err != nil {
		return fragutils.FragmentSeries{}, fmt.Errorf("failed to list fragments for incident: %w", err)
	}

	series := fragutils.NewFragmentSeries(fragments)
	series.TrimEnd(endTime)
	series.TrimStart(startTime)

	return series, nil
}

func (client *Client) putFragment(ctx context.Context, key fragkey.FragmentKey, fragment types.Fragment) error {
	logger := log.Ctx(ctx)

	media, err := client.kvsClient.GetMediaForFragment(ctx, aws.ToString(fragment.FragmentNumber))
	if err != nil {
		return fmt.Errorf("failed to get video data for fragment %q: %w", key, err)
	}

	metadata := fragarchive.FragmentMetadata{
		FragmentNumber:    aws.ToString(fragment.FragmentNumber),
		ProducerTimestamp: aws.ToTime(fragment.ProducerTimestamp),
		ServerTimestamp:   aws.ToTime(fragment.ServerTimestamp),
		Duration:          time.Duration(fragment.FragmentLengthInMilliseconds) * time.Millisecond,
		CameraUUID:        key.CameraUUID,
	}

	err = client.archive.PutFragment(ctx, key, media, fragment.FragmentSizeInBytes, metadata)
	if err != nil {
		return fmt.Errorf("failed to put fragment %q to archive: %w", key, err)
	}

	logger.Info().Stringer("fragment key", &key).Msgf("Put fragment media")

	return nil
}
