package fragarchive_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/smithy-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/lib/utils/aws/fake"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive"
	fragkey "github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive/key"
)

const (
	metadataDurationKey          = "x-amz-meta-duration-ms"
	metadataFragmentNumberKey    = "x-amz-meta-fragment-number"
	metadataProducerTimestampKey = "x-amz-meta-producer-timestamp-ms"
	metadataServerTimestampKey   = "x-amz-meta-server-timestamp-ms"
	metadataCameraUUIDKey        = "x-amz-meta-camera-uuid"
)

// very basic test to make sure the New function doesn't panic
func TestNew(t *testing.T) {
	s3Client := new(fake.S3Client)
	fragarchive.New(s3Client, "bucket")
}

func TestFragmentExists(t *testing.T) {
	bucketName := "test-bucket"
	s3Client := new(fake.S3Client)
	archive := fragarchive.New(s3Client, bucketName)

	testCases := []struct {
		name          string
		key           string
		mockErrReturn error
		expectError   bool
		expectExists  bool
	}{
		{
			name:          "fragment exists",
			key:           "test/key/1/1234567890.mkv",
			mockErrReturn: nil,
			expectError:   false,
			expectExists:  true,
		},
		{
			name:          "fragment does not exist",
			key:           "test/key/2/1234567890.mkv",
			mockErrReturn: &smithy.GenericAPIError{Code: "NotFound"},
			expectError:   false,
			expectExists:  false,
		},
		{
			name:          "error",
			key:           "test/key/3/1234567890.mkv",
			mockErrReturn: &smithy.GenericAPIError{Code: "SomeOtherError"},
			expectError:   true,
			expectExists:  false,
		},
	}

	s3Client.HeadObjectStub = func(ctx context.Context, params *s3.HeadObjectInput, optFns ...func(*s3.Options)) (*s3.HeadObjectOutput, error) {
		require.Equal(t, bucketName, *params.Bucket)
		for _, testCase := range testCases {
			if *params.Key == testCase.key {
				return nil, testCase.mockErrReturn
			}
		}
		require.FailNow(t, "unexpected key", "key: %s", *params.Key)
		return nil, nil
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			key, err := fragkey.NewFromString(testCase.key)
			require.NoError(t, err)
			exists, err := archive.FragmentExists(context.Background(), key)
			if testCase.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, testCase.expectExists, exists)
		})
	}
}

func TestPutFragment(t *testing.T) {
	bucketName := "test-bucket"
	s3Client := new(fake.S3Client)
	archive := fragarchive.New(s3Client, bucketName)

	testCases := []struct {
		name             string
		inputKey         string
		inputBodyStr     string
		inputMetadata    fragarchive.FragmentMetadata
		expectedMetadata map[string]string
		mockErrReturn    error
		expectError      bool
	}{
		{
			name:         "normal case",
			inputKey:     "test/camera/1/1234567890.mkv",
			inputBodyStr: "test body 1",
			inputMetadata: fragarchive.FragmentMetadata{
				Duration:          time.Second*9 + time.Millisecond*901,
				FragmentNumber:    "test-fragment-number-1",
				ProducerTimestamp: time.UnixMilli(1234567890123),
				ServerTimestamp:   time.UnixMilli(1234567890999),
				CameraUUID:        "test/camera/1",
			},
			expectedMetadata: map[string]string{
				metadataDurationKey:          "9901",
				metadataFragmentNumberKey:    "test-fragment-number-1",
				metadataProducerTimestampKey: "1234567890123",
				metadataServerTimestampKey:   "1234567890999",
				metadataCameraUUIDKey:        "test/camera/1",
			},
			mockErrReturn: nil,
			expectError:   false,
		},
		{
			name:         "failed put",
			inputKey:     "test/camera/2/1234567890.mkv",
			inputBodyStr: "test body 2",
			inputMetadata: fragarchive.FragmentMetadata{
				Duration:          time.Second*9 + time.Millisecond*902,
				FragmentNumber:    "test-fragment-number-2",
				ProducerTimestamp: time.UnixMilli(1234567890123),
				ServerTimestamp:   time.UnixMilli(1234567890999),
				CameraUUID:        "test/camera/2",
			},
			mockErrReturn: &smithy.GenericAPIError{Code: "SomeError"},
			expectError:   true,
		},
		{
			name:         "invalid metadata",
			inputKey:     "test/camera/3/1234567890.mkv",
			inputBodyStr: "test body 3",
			inputMetadata: fragarchive.FragmentMetadata{
				Duration:          time.Second * 100201,
				FragmentNumber:    "test-fragment-number-3",
				ProducerTimestamp: time.UnixMilli(1234567890123),
				ServerTimestamp:   time.UnixMilli(1234567890999),
			},
			expectedMetadata: map[string]string{
				metadataDurationKey:          "100201",
				metadataFragmentNumberKey:    "test-fragment-number-3",
				metadataProducerTimestampKey: "1234567890123",
				metadataServerTimestampKey:   "1234567890999",
			},
			mockErrReturn: nil,
			expectError:   true,
		},
		{
			name:         "invalid metadata (mismatched camera uuid)",
			inputKey:     "test/camera/4/1234567890.mkv",
			inputBodyStr: "test body 4",
			inputMetadata: fragarchive.FragmentMetadata{
				Duration:          time.Second*9 + time.Millisecond*904,
				FragmentNumber:    "test-fragment-number-4",
				ProducerTimestamp: time.UnixMilli(1234567890123),
				ServerTimestamp:   time.UnixMilli(1234567890999),
				CameraUUID:        "some different camera uuid",
			},
			mockErrReturn: nil,
			expectError:   true,
		},
		{
			name:         "invalid metadata (mismatched timestamp)",
			inputKey:     "test/camera/5/1234567890.mkv",
			inputBodyStr: "test body 5",
			inputMetadata: fragarchive.FragmentMetadata{
				Duration:          time.Second*9 + time.Millisecond*905,
				FragmentNumber:    "test-fragment-number-5",
				ProducerTimestamp: time.UnixMilli(9876543210123),
				ServerTimestamp:   time.UnixMilli(1234567890999),
				CameraUUID:        "test/camera/5",
			},
			mockErrReturn: nil,
			expectError:   true,
		},
	}

	s3Client.PutObjectStub = func(ctx context.Context, params *s3.PutObjectInput, optFns ...func(*s3.Options)) (*s3.PutObjectOutput, error) {
		for _, testCase := range testCases {
			if testCase.inputKey == *params.Key {
				bodyBytes, err := io.ReadAll(params.Body)
				require.NoError(t, err)

				assert.Equal(t, bucketName, *params.Bucket)
				assert.Equal(t, testCase.inputBodyStr, string(bodyBytes))

				if testCase.expectedMetadata != nil {
					assert.Equal(t, testCase.expectedMetadata, params.Metadata)
				}

				return nil, testCase.mockErrReturn
			}
		}
		require.FailNow(t, "unexpected key", "key: %s", *params.Key)
		return nil, nil
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			key, err := fragkey.NewFromString(testCase.inputKey)
			require.NoError(t, err)
			body := strings.NewReader(testCase.inputBodyStr)
			contentLength := int64(len(testCase.inputBodyStr))

			err = archive.PutFragment(context.Background(), key, body, contentLength, testCase.inputMetadata)
			if testCase.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestGetMetadata(t *testing.T) {
	ctx := context.Background()

	testBucketName := "bucket"
	testContentType := "video/x-matroska"
	testFragKey := fragkey.FragmentKey{
		CameraUUID:    "test-camera-uuid",
		UnixTimestamp: 1234567890,
		FileFormat:    fragkey.FormatMKV,
	}

	testCases := []struct {
		name               string
		fragKey            fragkey.FragmentKey
		s3HeadObjectOutput s3.HeadObjectOutput
		s3HeadObjectError  error
		expectedMetadata   fragarchive.FragmentMetadata
		expectError        bool
	}{
		{
			name:    "normal case",
			fragKey: testFragKey,
			s3HeadObjectOutput: s3.HeadObjectOutput{
				ContentLength: 1234,
				ContentType:   &testContentType,
				Metadata: map[string]string{
					"x-amz-meta-duration-ms":           "10000",
					"x-amz-meta-producer-timestamp-ms": "1234567890123",
					"x-amz-meta-server-timestamp-ms":   "1234567890123",
					"x-amz-meta-fragment-number":       "56913918398",
					"x-amz-meta-camera-uuid":           "test-camera-uuid",
				},
			},
			s3HeadObjectError: nil,
			expectedMetadata: fragarchive.FragmentMetadata{
				Duration:          10 * time.Second,
				FragmentNumber:    "56913918398",
				ProducerTimestamp: time.UnixMilli(1234567890123),
				ServerTimestamp:   time.UnixMilli(1234567890123),
				CameraUUID:        "test-camera-uuid",
			},
			expectError: false,
		},
		{
			name:    "partial metadata",
			fragKey: testFragKey,
			s3HeadObjectOutput: s3.HeadObjectOutput{
				ContentLength: 1234,
				ContentType:   &testContentType,
				Metadata: map[string]string{
					"x-amz-meta-duration-ms":           "10000",
					"x-amz-meta-producer-timestamp-ms": "1234567890123",
					"x-amz-meta-server-timestamp-ms":   "1234567890123",
				},
			},
			s3HeadObjectError: nil,
			expectedMetadata: fragarchive.FragmentMetadata{
				Duration:          10 * time.Second,
				ProducerTimestamp: time.UnixMilli(1234567890123),
				ServerTimestamp:   time.UnixMilli(1234567890123),
			},
			expectError: false,
		},
		{
			name:    "bad metadata",
			fragKey: testFragKey,
			s3HeadObjectOutput: s3.HeadObjectOutput{
				ContentLength: 1234,
				ContentType:   &testContentType,
				Metadata: map[string]string{
					"x-amz-meta-duration-ms":           "10000",
					"x-amz-meta-producer-timestamp-ms": "not a timestamp",
					"x-amz-meta-server-timestamp-ms":   "1234567890123",
					"x-amz-meta-fragment-number":       "56913918398",
					"x-amz-meta-camera-uuid":           "test-camera-uuid",
				},
			},
			s3HeadObjectError: nil,
			expectError:       true,
		},
		{
			name:               "s3 error",
			fragKey:            testFragKey,
			s3HeadObjectOutput: s3.HeadObjectOutput{},
			s3HeadObjectError:  fmt.Errorf("test error"),
			expectError:        true,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			s3Client := new(fake.S3Client)
			archive := fragarchive.New(s3Client, testBucketName)

			s3Client.HeadObjectReturns(&testCase.s3HeadObjectOutput, testCase.s3HeadObjectError)

			metadata, err := archive.GetMetadata(ctx, testCase.fragKey)

			require.Equal(t, 1, s3Client.HeadObjectCallCount())
			_, input, _ := s3Client.HeadObjectArgsForCall(0)

			assert.Equal(t, testBucketName, *input.Bucket)
			assert.Equal(t, testCase.fragKey.String(), *input.Key)

			if testCase.expectError {
				assert.Error(t, err, "should error")
			} else {
				assert.NoError(t, err, "should not error")
				assert.Equal(t, testCase.expectedMetadata, metadata, "metadata should match")
			}
		})
	}
}

func TestGetFragment(t *testing.T) {
	s3Client := new(fake.S3Client)
	bucketName := "test-bucket"
	ctx := context.Background()

	archive := fragarchive.New(s3Client, bucketName)

	testCases := []struct {
		name            string
		key             string
		mockOutput      s3.GetObjectOutput
		mockError       error
		expectedPayload []byte
		expectedMeta    fragarchive.FragmentMetadata
		expectError     bool
	}{
		{
			name: "normal test case",
			key:  "test/camera/id/1234567890.mkv",
			mockOutput: s3.GetObjectOutput{
				Body: io.NopCloser(strings.NewReader("test")),
				Metadata: map[string]string{
					"x-amz-meta-duration-ms":           "10000",
					"x-amz-meta-fragment-number":       "1234567890",
					"x-amz-meta-producer-timestamp-ms": "1234567890000",
					"x-amz-meta-server-timestamp-ms":   "1234567890500",
					"x-amz-meta-camera-uuid":           "test/camera/id",
				},
			},
			mockError:       nil,
			expectedPayload: []byte("test"),
			expectedMeta: fragarchive.FragmentMetadata{
				Duration:          10 * time.Second,
				FragmentNumber:    "1234567890",
				ProducerTimestamp: time.Unix(1234567890, 0),
				ServerTimestamp:   time.Unix(1234567890, 500000000),
				CameraUUID:        "test/camera/id",
			},
			expectError: false,
		},
		{
			name: "missing metadata",
			key:  "test/camera/id/2234567890.mkv",
			mockOutput: s3.GetObjectOutput{
				Body: io.NopCloser(strings.NewReader("test")),
			},
			mockError:       nil,
			expectError:     false,
			expectedPayload: []byte("test"),
			expectedMeta:    fragarchive.FragmentMetadata{},
		},
		{
			name:        "no object exists error",
			key:         "test/camera/id/3334567890.mkv",
			mockOutput:  s3.GetObjectOutput{},
			mockError:   errors.New("NoSuchKey: The specified key does not exist"),
			expectError: true,
		},
		{
			name: "bad metadata",
			key:  "test/camera/id/4444567890.mkv",
			mockOutput: s3.GetObjectOutput{
				Body: io.NopCloser(strings.NewReader("test")),
				Metadata: map[string]string{
					"x-amz-meta-duration-ms":           "not a number",
					"x-amz-meta-fragment-number":       "1234506789",
					"x-amz-meta-producer-timestamp-ms": "1234567890000",
					"x-amz-meta-server-timestamp-ms":   "1234567890500",
					"x-amz-meta-camera-uuid":           "test/camera/id",
				},
			},
			mockError:   nil,
			expectError: true,
		},
	}

	s3Client.GetObjectStub = func(ctx context.Context, params *s3.GetObjectInput, optFns ...func(*s3.Options)) (*s3.GetObjectOutput, error) {
		require.Equal(t, bucketName, *params.Bucket)
		for _, testCase := range testCases {
			if testCase.key == *params.Key {
				return &testCase.mockOutput, testCase.mockError
			}
		}

		assert.FailNow(t, "unexpected key", "key %s not found in test cases", *params.Key)
		return nil, nil
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			fragKey, err := fragkey.NewFromString(testCase.key)
			require.NoError(t, err)

			payload, meta, err := archive.GetFragment(ctx, fragKey)
			if testCase.expectError {
				assert.Error(t, err, testCase.name)
				return
			}

			assert.NoError(t, err, testCase.name)

			assert.Equal(t, testCase.expectedPayload, payload, testCase.name)
			assert.Equal(t, testCase.expectedMeta, meta, testCase.name)
		})
	}
}
