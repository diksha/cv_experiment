package fragarchive_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive"
)

func TestParseFragmentMetadata(t *testing.T) {
	testCases := []struct {
		name             string
		inputMetadata    map[string]string
		expectError      bool
		expectedMetadata fragarchive.FragmentMetadata
	}{
		{
			name: "normal case",
			inputMetadata: map[string]string{
				"Content-Type":               "application/octet-stream",
				metadataDurationKey:          "9998",
				metadataCameraUUIDKey:        "michaels/tracy/0002/cha",
				metadataFragmentNumberKey:    "91343852336970796563495554524102405879022788815",
				metadataProducerTimestampKey: "1682342490978",
				metadataServerTimestampKey:   "1682342511498",
			},
			expectError: false,
			expectedMetadata: fragarchive.FragmentMetadata{
				Duration:          time.Millisecond * 9998,
				FragmentNumber:    "91343852336970796563495554524102405879022788815",
				ProducerTimestamp: time.UnixMilli(1682342490978),
				ServerTimestamp:   time.UnixMilli(1682342511498),
				CameraUUID:        "michaels/tracy/0002/cha",
			},
		},
		{
			name: "non-int duration",
			inputMetadata: map[string]string{
				"Content-Type":               "application/octet-stream",
				metadataDurationKey:          "9998.999",
				metadataCameraUUIDKey:        "michaels/tracy/0002/cha",
				metadataFragmentNumberKey:    "91343852336970796563495554524102405879022788815",
				metadataProducerTimestampKey: "1682342490978",
				metadataServerTimestampKey:   "1682342511498",
			},
			expectError: true,
		},
		{
			name: "non-int producer timestamp",
			inputMetadata: map[string]string{
				"Content-Type":               "application/octet-stream",
				metadataDurationKey:          "9998",
				metadataCameraUUIDKey:        "michaels/tracy/0002/cha",
				metadataFragmentNumberKey:    "91343852336970796563495554524102405879022788815",
				metadataProducerTimestampKey: "1682342490978.999",
				metadataServerTimestampKey:   "1682342511498",
			},
			expectError: true,
		},
		{
			name: "non-int server timestamp",
			inputMetadata: map[string]string{
				"Content-Type":               "application/octet-stream",
				metadataDurationKey:          "9998",
				metadataCameraUUIDKey:        "michaels/tracy/0002/cha",
				metadataFragmentNumberKey:    "91343852336970796563495554524102405879022788815",
				metadataProducerTimestampKey: "1682342490978",
				metadataServerTimestampKey:   "1682342511498.999",
			},
			expectError: true,
		},
		{
			name:             "empty input",
			inputMetadata:    map[string]string{},
			expectError:      false,
			expectedMetadata: fragarchive.FragmentMetadata{},
		},
		{
			name: "missing fragment number",
			inputMetadata: map[string]string{
				"Content-Type":               "application/octet-stream",
				metadataDurationKey:          "9998",
				metadataCameraUUIDKey:        "michaels/tracy/0002/cha",
				metadataProducerTimestampKey: "1682342490978",
				metadataServerTimestampKey:   "1682342511498",
			},
			expectError: false,
			expectedMetadata: fragarchive.FragmentMetadata{
				Duration:          time.Millisecond * 9998,
				ProducerTimestamp: time.UnixMilli(1682342490978),
				ServerTimestamp:   time.UnixMilli(1682342511498),
				CameraUUID:        "michaels/tracy/0002/cha",
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			metadata, err := fragarchive.ParseFragmentMetadata(testCase.inputMetadata)
			if testCase.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, testCase.expectedMetadata, metadata)
			}
		})
	}
}
