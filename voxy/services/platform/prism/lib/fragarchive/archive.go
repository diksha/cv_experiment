// Package fragarchive provides an interface for manipulating the archive of
// fragments which make up incident video data.
package fragarchive

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/smithy-go"
	"github.com/davecgh/go-spew/spew"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	fragkey "github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive/key"
)

// S3API is the subset of the S3 client API used by the fragment archive
type S3API interface {
	HeadObject(ctx context.Context, params *s3.HeadObjectInput, optFns ...func(*s3.Options)) (*s3.HeadObjectOutput, error)
	PutObject(ctx context.Context, params *s3.PutObjectInput, optFns ...func(*s3.Options)) (*s3.PutObjectOutput, error)
	GetObject(ctx context.Context, params *s3.GetObjectInput, optFns ...func(*s3.Options)) (*s3.GetObjectOutput, error)
}

var _ S3API = (*s3.Client)(nil)

// Client is the struct for interacting with the fragment archive
type Client struct {
	s3     S3API
	bucket string
}

// New creates a new fragment archive client
func New(s3 S3API, bucket string) *Client {
	return &Client{
		s3:     s3,
		bucket: bucket,
	}
}

// FragmentExists returns true iff the provided key references a fragment in the archive
func (c *Client) FragmentExists(ctx context.Context, key fragkey.FragmentKey) (bool, error) {
	stringKey := key.String()
	input := &s3.HeadObjectInput{
		Bucket: &c.bucket,
		Key:    &stringKey,
	}

	_, err := c.s3.HeadObject(ctx, input)
	if err != nil {
		var apiErr smithy.APIError
		if errors.As(err, &apiErr) {
			if apiErr.ErrorCode() == "NotFound" {
				return false, nil
			}
		}
		return false, fmt.Errorf("S3 head object request failed: %w", err)
	}

	return true, nil
}

// PutFragment uploads a fragment to the archive
func (c *Client) PutFragment(ctx context.Context, key fragkey.FragmentKey, body io.Reader, contentLength int64, metadata FragmentMetadata) error {
	logger := log.Ctx(ctx)

	err := metadata.validate(ctx)
	if err != nil {
		loggableMetadata := zerolog.Dict().
			Dur("duration", metadata.Duration).
			Str("fragment number", metadata.FragmentNumber).
			Time("producer timestamp", metadata.ProducerTimestamp).
			Time("server timestamp", metadata.ServerTimestamp).
			Str("camera uuid", metadata.CameraUUID)

		logger.Debug().
			Str("fragment key", key.String()).
			Dict("metadata", loggableMetadata).
			Msg("invalid fragment metadata")

		return fmt.Errorf("invalid fragment metadata: %w", err)
	}

	if metadata.CameraUUID != key.CameraUUID {
		return fmt.Errorf("fragment metadata CameraUUID (%s) does not match key CameraUUID (%s)", metadata.CameraUUID, key.CameraUUID)
	}

	if metadata.ProducerTimestamp.Unix() != key.UnixTimestamp {
		return fmt.Errorf("fragment metadata ProducerTimestamp (%d) does not match key UnixTimestamp (%d)", metadata.ProducerTimestamp.UnixMilli(), key.UnixTimestamp)
	}

	stringKey := key.String()
	input := &s3.PutObjectInput{
		Bucket:        &c.bucket,
		Key:           &stringKey,
		Body:          body,
		ContentLength: contentLength,
		Metadata:      metadata.toMap(),
	}

	_, err = c.s3.PutObject(ctx, input)
	if err != nil {
		logger.Debug().
			Err(err).
			Str("PutObjectInput.Bucket", c.bucket).
			Str("PutObjectInput.Key", stringKey).
			Int64("PutObjectInput.ContentLength", contentLength).
			Str("PutObjectInput.Metadata", spew.Sdump(input.Metadata)).
			Msg("failed call to S3:PutObject")

		return fmt.Errorf("S3:PutObject request failed: %w", err)
	}

	return nil
}

// GetMetadata retrieves the metadata for a fragment from the archive
func (c *Client) GetMetadata(ctx context.Context, key fragkey.FragmentKey) (FragmentMetadata, error) {
	logger := log.Ctx(ctx)

	stringKey := key.String()
	input := &s3.HeadObjectInput{
		Bucket: &c.bucket,
		Key:    &stringKey,
	}

	output, err := c.s3.HeadObject(ctx, input)
	if err != nil {
		logger.Debug().
			Err(err).
			Str("HeadObjectInput.Bucket", c.bucket).
			Str("HeadObjectInput.Key", stringKey).
			Msg("failed call to S3:HeadObject")

		return FragmentMetadata{}, fmt.Errorf("S3 head object request failed: %w", err)
	}

	metadata, err := ParseFragmentMetadata(output.Metadata)
	if err != nil {
		logger.Debug().
			Err(err).
			Interface("HeadObjectOutput.Metadata", output.Metadata).
			Msg("failed to parse fragment metadata")

		return FragmentMetadata{}, fmt.Errorf("failed to parse fragment metadata: %w", err)
	}

	return metadata, nil
}

// GetFragment retrieves a fragment from the archive
func (c *Client) GetFragment(ctx context.Context, key fragkey.FragmentKey) ([]byte, FragmentMetadata, error) {
	stringKey := key.String()
	input := &s3.GetObjectInput{
		Bucket: &c.bucket,
		Key:    &stringKey,
	}

	output, err := c.s3.GetObject(ctx, input)
	if err != nil {
		log.Debug().
			Err(err).
			Str("GetObjectInput.Bucket", c.bucket).
			Str("GetObjectInput.Key", stringKey).
			Msg("failed call to S3:GetObject")

		return nil, FragmentMetadata{}, fmt.Errorf("S3:GetObject request failed: %w", err)
	}

	metadata, err := ParseFragmentMetadata(output.Metadata)
	if err != nil {
		log.Debug().
			Err(err).
			Interface("GetObjectOutput.Metadata", output.Metadata).
			Msg("failed call to S3:GetObject")

		return nil, FragmentMetadata{}, fmt.Errorf("invalid fragment metadata: %w", err)
	}

	outputBytes, err := io.ReadAll(output.Body)
	if err != nil {
		return nil, FragmentMetadata{}, fmt.Errorf("failed to read fragment body: %w", err)
	}

	return outputBytes, metadata, nil
}

// RangeQuery is used to query the archive for fragments which overlap with the provided time range
type RangeQuery struct {
	StartTime  time.Time
	EndTime    time.Time
	CameraUUID string
}

// GetFragmentKeysInRange returns the keys of all fragments in the archive which
// overlap with the provided time range.
func (c *Client) GetFragmentKeysInRange(ctx context.Context, query RangeQuery) ([]fragkey.FragmentKey, error) {
	return []fragkey.FragmentKey{}, fmt.Errorf("GetFragmentKeysInRange not implemented")
}
