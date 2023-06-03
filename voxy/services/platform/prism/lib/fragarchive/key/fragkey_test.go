package fragkey_test

import (
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/kinesisvideoarchivedmedia/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fragkey "github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive/key"
)

const testCameraUUID = "wesco/reno/002"

func TestNewFromTimestamp(t *testing.T) {
	timestamp := time.Unix(1678488133, 0)
	expectedKey := fragkey.FragmentKey{
		CameraUUID:    "wesco/reno/002",
		UnixTimestamp: 1678488133,
		FileFormat:    fragkey.FormatMKV,
	}

	key, err := fragkey.NewFromTimestamp(testCameraUUID, timestamp, fragkey.FormatMKV)
	assert.NoError(t, err, "should not error on good timestamp")
	assert.Equal(t, expectedKey, key)

	// Invalid Timestamp
	timestamp = time.Unix(11678488133, 0)
	require.Error(t, fragkey.ValidateTimestamp(timestamp), "should error on out of bounds timestamp")

	_, err = fragkey.NewFromTimestamp(testCameraUUID, timestamp, fragkey.FormatMKV)
	assert.Error(t, err, "should error on out of bounds timestamp")
}

func TestNewFromFragment(t *testing.T) {
	fragNumber := "129810938012930981"
	producerTimestamp := time.Unix(1678488133, 999999999)
	serverTimestamp := time.Unix(1678488221, 0)
	fragment := types.Fragment{
		FragmentLengthInMilliseconds: 10000,
		FragmentNumber:               &fragNumber,
		FragmentSizeInBytes:          12212,
		ProducerTimestamp:            &producerTimestamp,
		ServerTimestamp:              &serverTimestamp,
	}

	expectedKey := fragkey.FragmentKey{
		CameraUUID:    "wesco/reno/002",
		UnixTimestamp: 1678488133,
		FileFormat:    fragkey.FormatMKV,
	}

	key, err := fragkey.NewFromFragment(testCameraUUID, fragment, fragkey.FormatMKV)
	assert.NoError(t, err)
	assert.Equal(t, expectedKey, key)
}

func TestNewFromString(t *testing.T) {
	goodKeys := map[string]fragkey.FragmentKey{
		"wesco/reno/002/1678488133.mkv": {
			CameraUUID:    "wesco/reno/002",
			UnixTimestamp: 1678488133,
			FileFormat:    fragkey.FormatMKV,
		},
		"wesco.reno.002/1678488133.mkv": {
			CameraUUID:    "wesco.reno.002",
			UnixTimestamp: 1678488133,
			FileFormat:    fragkey.FormatMKV,
		},
	}

	badKeys := []string{
		"wesco/reno/002/1678488133.",
		"wesco/reno/002/1678488133/",
		"wesco/reno/002/1678488133",
		"wesco/reno/002/1678488133.badextension",
		"wesco/reno/002/10000.mkv",
	}

	for goodKey, expectedKey := range goodKeys {
		key, err := fragkey.NewFromString(goodKey)
		assert.NoError(t, err)
		assert.Equal(t, expectedKey, key)
	}

	for _, badKey := range badKeys {
		_, err := fragkey.NewFromString(badKey)
		assert.Error(t, err)
	}
}

func TestString(t *testing.T) {
	expectedStr := "wesco/reno/002/1678488133.mkv"
	key := fragkey.FragmentKey{
		CameraUUID:    "wesco/reno/002",
		UnixTimestamp: 1678488133,
		FileFormat:    fragkey.FormatMKV,
	}

	assert.Equal(t, expectedStr, key.String())
}

// Test Validate Timestamp
func TestValidateTimestamp(t *testing.T) {
	timestamp := time.Unix(1678488133, 0)
	err := fragkey.ValidateTimestamp(timestamp)
	assert.NoError(t, err, "should not error on good timestamp")

	// Invalid Timestamp (too large)
	timestamp = time.Unix(11678488133, 0)
	err = fragkey.ValidateTimestamp(timestamp)
	assert.Error(t, err, "should error on out of bounds timestamp")

	// Invalid Timestamp (too small)
	timestamp = time.Unix(978488133, 0)
	err = fragkey.ValidateTimestamp(timestamp)
	assert.Error(t, err, "should error on out of bounds timestamp")
}
