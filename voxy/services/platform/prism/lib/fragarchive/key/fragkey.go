// Package fragkey provides helper functions for referencing fragments in the S3 bucket
package fragkey

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/kinesisvideoarchivedmedia/types"
)

// FileFormat represents the video type of the fragment
type FileFormat string

// FormatMKV represents a .mvk file type
const FormatMKV FileFormat = "mkv"

var allFileFormats = [...]FileFormat{FormatMKV}

// FragmentKey uniquely identifies a fragment in S3
type FragmentKey struct {
	CameraUUID    string
	UnixTimestamp int64
	FileFormat    FileFormat
}

/*
Keep timestamps to 10 digits for key sorting
If we find some where we need data from before 2001 can add in zero-padding
*/
const minTimestamp = 1000000000 // Sep 09 2001 01:46:40 GMT+0000
const maxTimestamp = 9999999999 // Nov 20 2286 17:46:39 GMT+0000

// ValidateTimestamp ensures that the timestamp is within the range of valid timestamps
func ValidateTimestamp(timestamp time.Time) error {
	return validateTimestampUnix(timestamp.Unix())
}

func validateTimestampUnix(unixTimestamp int64) error {
	if unixTimestamp < minTimestamp {
		return fmt.Errorf("invalid unix timestamp %q must occur after %q", time.Unix(unixTimestamp, 0), time.Unix(minTimestamp, 0))
	}

	if unixTimestamp > maxTimestamp {
		return fmt.Errorf("invalid unix timestamp %q must occur before %q", time.Unix(unixTimestamp, 0), time.Unix(maxTimestamp, 0))
	}

	return nil
}

func validateFormat(fileFormat FileFormat) error {
	for _, supportedFormat := range allFileFormats {
		if fileFormat == supportedFormat {
			return nil
		}
	}

	return fmt.Errorf("unsupported fileformat %q", fileFormat)
}

// NewFromUnixTimestamp creates a new fragment key from the cameraUUID and Unix timestamp (in seconds)
// The creation of a FragmentKey does not guarantee existence of such fragment
func NewFromUnixTimestamp(cameraUUID string, unixTimestampSeconds int64, format FileFormat) (FragmentKey, error) {
	if err := validateFormat(format); err != nil {
		return FragmentKey{}, err
	}

	if err := validateTimestampUnix(unixTimestampSeconds); err != nil {
		return FragmentKey{}, err
	}

	return FragmentKey{
		CameraUUID:    cameraUUID,
		UnixTimestamp: unixTimestampSeconds,
		FileFormat:    format,
	}, nil
}

// NewFromTimestamp creates a new fragment key from the cameraUUID and timestamp
// The creation of a FragmentKey does not guarantee existence of such fragment
func NewFromTimestamp(cameraUUID string, timestamp time.Time, format FileFormat) (FragmentKey, error) {
	return NewFromUnixTimestamp(cameraUUID, timestamp.Unix(), format)
}

// NewFromFragment creates a new fragment key from the cameraUUID and fragment from Kinesis Video Archived Media
// The creation of a FragmentKey does not guarantee existence of such fragment in S3
func NewFromFragment(cameraUUID string, fragment types.Fragment, format FileFormat) (FragmentKey, error) {
	return NewFromUnixTimestamp(cameraUUID, fragment.ProducerTimestamp.Unix(), format)
}

// NewFromString creates a new fragment key from a properly formatted string representation of the key
// The creation of a FragmentKey does not guarantee existence of such fragment in S3
func NewFromString(key string) (FragmentKey, error) {
	firstSplit := strings.LastIndex(key, "/")
	secondSplit := strings.LastIndex(key, ".")

	if firstSplit == -1 || secondSplit == -1 || firstSplit > secondSplit || secondSplit == len(key)-1 {
		return FragmentKey{}, fmt.Errorf("failed to parse string %q to a fragment key", key)
	}

	cameraUUID, timestampStr, formatStr := key[:firstSplit], key[firstSplit+1:secondSplit], key[secondSplit+1:]
	format := FileFormat(formatStr)

	timestampInt, err := strconv.ParseInt(timestampStr, 10, 64)
	if err != nil {
		return FragmentKey{}, fmt.Errorf("failed to parse string %q to int64: %w", timestampStr, err)
	}

	return NewFromUnixTimestamp(cameraUUID, timestampInt, format)
}

func (key *FragmentKey) String() string {
	return fmt.Sprintf("%v/%v.%v", key.CameraUUID, key.UnixTimestamp, key.FileFormat)
}
