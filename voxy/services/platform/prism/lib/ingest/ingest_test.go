package ingest_test

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-lambda-go/events"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideoarchivedmedia"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideoarchivedmedia/types"
	"github.com/aws/aws-sdk-go-v2/service/s3"

	"github.com/aws/smithy-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/lib/utils/aws/fake"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive"
	fragkey "github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive/key"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/incident"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/ingest"
)

func stubHeadObject(fakeS3Client *fake.S3Client, keysReturnFound, keysReturnError []string) {
	fakeS3Client.HeadObjectStub = func(ctx context.Context, params *s3.HeadObjectInput, optFns ...func(*s3.Options)) (*s3.HeadObjectOutput, error) {
		for _, key := range keysReturnFound {
			if *params.Key == key {
				return &s3.HeadObjectOutput{}, nil
			}
		}

		for _, key := range keysReturnError {
			if *params.Key == key {
				return nil, &smithy.GenericAPIError{Code: "TransientFailure"}
			}
		}

		return nil, &smithy.GenericAPIError{Code: "NotFound"}
	}
}

func stubGetDataEndpoint(fakeKVClient *fake.KinesisVideoClient, dataEndpoint string) {
	fakeKVClient.GetDataEndpointStub = func(ctx context.Context, input *kinesisvideo.GetDataEndpointInput, opt ...func(*kinesisvideo.Options)) (*kinesisvideo.GetDataEndpointOutput, error) {
		streamNameDefined := input.StreamName != nil
		streamARNDefined := input.StreamARN != nil

		if streamNameDefined == streamARNDefined {
			return nil, &smithy.GenericAPIError{Code: "BadRequestException"}
		}

		return &kinesisvideo.GetDataEndpointOutput{
			DataEndpoint: &dataEndpoint,
		}, nil
	}
}

func stubGetMediaForFragmentList(fakeKVAMClient *fake.KinesisVideoArchivedMediaClient) {
	fakeKVAMClient.GetMediaForFragmentListStub = func(_ context.Context, input *kinesisvideoarchivedmedia.GetMediaForFragmentListInput, _ ...func(*kinesisvideoarchivedmedia.Options)) (*kinesisvideoarchivedmedia.GetMediaForFragmentListOutput, error) {
		var fragmentNumbersString string
		for _, fragmentNumber := range input.Fragments {
			fragmentNumbersString += fragmentNumber
		}

		reader := io.NopCloser(strings.NewReader(fragmentNumbersString))
		contentType := "string"
		return &kinesisvideoarchivedmedia.GetMediaForFragmentListOutput{
			ContentType: &contentType,
			Payload:     reader,
		}, nil
	}
}

func stubListFragments(fakeKVAMClient *fake.KinesisVideoArchivedMediaClient, maxResults int) {
	fakeKVAMClient.ListFragmentsStub = func(ctx context.Context, params *kinesisvideoarchivedmedia.ListFragmentsInput, optFns ...func(*kinesisvideoarchivedmedia.Options)) (*kinesisvideoarchivedmedia.ListFragmentsOutput, error) {
		queryStartTime := *params.FragmentSelector.TimestampRange.StartTimestamp
		queryEndTime := *params.FragmentSelector.TimestampRange.EndTimestamp
		genStartTime := queryStartTime.Add(-time.Second * 10).Truncate(10 * time.Second)
		genEndTime := queryEndTime.Add(time.Second * 10).Truncate(10 * time.Second)

		callNumber := 0
		if params.NextToken != nil {
			callNumber, _ = strconv.Atoi(*params.NextToken)
		}
		resultsToSkip := callNumber * maxResults

		var nextToken *string
		// filter out the fragments which are not within the selected time range
		var filteredFragments []types.Fragment
		for _, fragment := range generateFragments(genStartTime, genEndTime) {
			inRange := !fragment.ProducerTimestamp.Before(queryStartTime) && !fragment.ProducerTimestamp.After(queryEndTime)

			if inRange {
				resultsToSkip--
				if resultsToSkip < 0 {
					filteredFragments = append(filteredFragments, fragment)
				}

				if resultsToSkip == -maxResults {
					nextToken = new(string)
					*nextToken = strconv.Itoa(callNumber + 1)
					break
				}
			}
		}

		return &kinesisvideoarchivedmedia.ListFragmentsOutput{
			Fragments: filteredFragments,
			NextToken: nextToken,
		}, nil
	}
}

func fragStartTimeToFragNumber(startTime time.Time) string {
	return startTime.Format("20060102150405")
}

func generateFragments(startTime time.Time, endTime time.Time) []types.Fragment {
	var fragments []types.Fragment
	for i := startTime; i.Before(endTime); i = i.Add(time.Second * 10) {
		fragment := types.Fragment{
			FragmentNumber:               new(string),
			FragmentLengthInMilliseconds: int64(10000),
			ProducerTimestamp:            new(time.Time),
			ServerTimestamp:              new(time.Time),
		}
		*fragment.FragmentNumber = fragStartTimeToFragNumber(i)
		fragment.FragmentSizeInBytes = int64(len(*fragment.FragmentNumber))
		*fragment.ProducerTimestamp = i
		*fragment.ServerTimestamp = i.Add(time.Second * 10)
		fragments = append(fragments, fragment)
	}
	return fragments
}

type testSetup struct {
	query struct {
		cameraUUID string
		startTime  time.Time
		endTime    time.Time
		streamARN  string
	}

	s3 struct {
		bucketName              string
		storedKeys              []string
		keysThatReturnHeadError []string
	}
	kvs struct {
		dataEndpoint string
	}
	kvam struct {
		maxResultsPerListFragmentCall int
	}
}

func (setup *testSetup) newKVAMClient() *fake.KinesisVideoArchivedMediaClient {
	client := new(fake.KinesisVideoArchivedMediaClient)
	stubGetMediaForFragmentList(client)
	stubListFragments(client, setup.kvam.maxResultsPerListFragmentCall)
	return client
}

func (setup *testSetup) newKVSClient() *fake.KinesisVideoClient {
	client := new(fake.KinesisVideoClient)
	stubGetDataEndpoint(client, setup.kvs.dataEndpoint)
	return client
}

func (setup *testSetup) newS3Client() *fake.S3Client {
	client := new(fake.S3Client)
	stubHeadObject(client, setup.s3.storedKeys, setup.s3.keysThatReturnHeadError)
	return client
}

func (setup *testSetup) allS3KeysInRange(t *testing.T) []string {
	var keys []string
	start := setup.query.startTime.Truncate(10 * time.Second)
	lastStartTime := setup.query.endTime.Add(-10 * time.Second).Truncate(1 * time.Second)
	t.Logf("i: %v", lastStartTime.UnixMilli())
	for i := start; ; i = i.Add(10 * time.Second) {
		key, err := fragkey.NewFromTimestamp(setup.query.cameraUUID, i, fragkey.FormatMKV)
		require.NoError(t, err)
		keys = append(keys, key.String())
		t.Logf("i: %v", i.UnixMilli())
		if i.After(lastStartTime) {
			break
		}
	}

	return keys
}

func (setup *testSetup) getKeysToPut(t *testing.T) []string {
	allKeys := setup.allS3KeysInRange(t)

	storedKeysMap := make(map[string]bool)
	for _, key := range setup.s3.storedKeys {
		storedKeysMap[key] = true
	}

	var keysToPut []string
	for _, key := range allKeys {
		if _, ok := storedKeysMap[key]; !ok {
			keysToPut = append(keysToPut, key)
		}
	}

	return keysToPut
}

func (setup *testSetup) makeS3HeadObjectAssertions(t *testing.T, s3Client *fake.S3Client) {
	allKeys := setup.allS3KeysInRange(t)
	require.Len(t, allKeys, s3Client.HeadObjectCallCount())

	allKeysMap := make(map[string]bool)
	for _, key := range allKeys {
		allKeysMap[key] = false
	}

	for i := 0; i < len(allKeys); i++ {
		_, input, _ := s3Client.HeadObjectArgsForCall(i)

		assert.Equal(t, setup.s3.bucketName, *input.Bucket)
		assert.Contains(t, allKeysMap, *input.Key)
		assert.False(t, allKeysMap[*input.Key], "duplicate key in s3Client.HeadObjectCallArgs")
		allKeysMap[*input.Key] = true

		s3Client.HeadObjectArgsForCall(i)
	}
}

func (setup *testSetup) makeS3PutObjectAssertions(t *testing.T, s3Client *fake.S3Client) {
	keysToPut := setup.getKeysToPut(t)
	keysToPutMap := make(map[string]bool)
	for _, key := range keysToPut {
		keysToPutMap[key] = false
	}

	require.Len(t, keysToPutMap, s3Client.PutObjectCallCount())

	for i := 0; i < len(keysToPutMap); i++ {
		_, input, _ := s3Client.PutObjectArgsForCall(i)

		bodyBytes, err := io.ReadAll(input.Body)
		require.NoError(t, err, "failed to read body")

		metadata, err := fragarchive.ParseFragmentMetadata(input.Metadata)
		require.NoError(t, err, "failed to parse metadata")

		keyString := *input.Key
		key, err := fragkey.NewFromString(keyString)
		require.NoError(t, err, "failed to parse key")

		body := string(bodyBytes)
		bodyLength := int64(len(bodyBytes))
		contentLength := input.ContentLength

		assert.Equal(t, setup.s3.bucketName, *input.Bucket)

		// validate that metadata matches the other values
		assert.Equal(t, metadata.FragmentNumber, body)
		assert.Equal(t, metadata.ProducerTimestamp.Unix(), key.UnixTimestamp)
		assert.Equal(t, metadata.CameraUUID, setup.query.cameraUUID)
		assert.Equal(t, bodyLength, contentLength)

		// assert all expected keys appear
		assert.Contains(t, keysToPutMap, *input.Key)
		assert.False(t, keysToPutMap[*input.Key], "duplicate key in s3Client.PutObjectCallArgs")
		keysToPutMap[*input.Key] = true
	}
}

func (setup *testSetup) makeKVAMGetMediaForFragmentListAssertions(t *testing.T, kvamClient *fake.KinesisVideoArchivedMediaClient) {
	keysToPut := setup.getKeysToPut(t)
	fragmentNumbers := make(map[string]bool)
	for _, keyStr := range keysToPut {
		key, err := fragkey.NewFromString(keyStr)
		require.NoError(t, err, "failed to parse key")
		fragmentNumber := fragStartTimeToFragNumber(time.Unix(key.UnixTimestamp, 0))
		fragmentNumbers[fragmentNumber] = false
	}

	require.Len(t, fragmentNumbers, kvamClient.GetMediaForFragmentListCallCount())
	for i := 0; i < len(fragmentNumbers); i++ {
		_, input, _ := kvamClient.GetMediaForFragmentListArgsForCall(i)
		// Assert that stream ID is correct
		assert.Equal(t, setup.query.streamARN, *input.StreamARN)
		assert.Nil(t, input.StreamName)

		require.Len(t, input.Fragments, 1)
		assert.Contains(t, fragmentNumbers, input.Fragments[0])
		assert.False(t, fragmentNumbers[input.Fragments[0]], "duplicate key in kvamClient.GetMediaForFragmentListCallArgs")
		fragmentNumbers[input.Fragments[0]] = true
	}
}

func TestIngestSQSEvent(t *testing.T) {
	ctx := context.Background()

	test := testSetup{}
	test.query.cameraUUID = "americold/ontario/0104/cha"
	test.query.startTime = time.UnixMilli(1682628844749 - 3000)
	test.query.endTime = time.UnixMilli(1682628854749 + 6000)
	test.query.streamARN = "arn:aws:kinesisvideo:us-west-2:115099983231:stream/walk-test-stream/1680811789506"

	t.Log(test.query.startTime.Unix())
	t.Log(test.query.endTime.Unix())

	// americold/ontario/0104/cha/1682628840.mkv
	// americold/ontario/0104/cha/1682628850.mkv
	// americold/ontario/0104/cha/1682628860.mkv
	t.Log(test.allS3KeysInRange(t))

	test.s3.bucketName = "test_bucket_name"
	test.s3.storedKeys = []string{}
	test.s3.keysThatReturnHeadError = []string{}

	test.kvs.dataEndpoint = "https://kinesisvideo.us-east-1.amazonaws.com"

	test.kvam.maxResultsPerListFragmentCall = 3

	testData, err := os.ReadFile("testdata/sample_sqs_event.json")
	require.NoError(t, err)

	event := events.SQSEvent{}
	err = json.Unmarshal(testData, &event)
	require.NoError(t, err)

	s3Client := test.newS3Client()
	kvsClient := test.newKVSClient()
	kvamClient := test.newKVAMClient()
	archiveClient := fragarchive.New(s3Client, test.s3.bucketName)

	ingestClient := ingest.Client{
		ArchiveClient: archiveClient,
		KVAMClient:    kvamClient,
		KVClient:      kvsClient,
	}

	err = ingestClient.HandleSQSEvent(ctx, event)
	assert.NoError(t, err)

	// log all calls to s3Client.HeadObject
	for i := 0; i < s3Client.HeadObjectCallCount(); i++ {
		_, input, _ := s3Client.HeadObjectArgsForCall(i)
		t.Logf("s3Client.HeadObject(%v)", *input.Key)
	}

	test.makeS3HeadObjectAssertions(t, s3Client)
	test.makeS3PutObjectAssertions(t, s3Client)
	test.makeKVAMGetMediaForFragmentListAssertions(t, kvamClient)
}

/*
Test Description

Times denoted by three digits which are to replace the xxx in
the unix timestamp 1672600xxx

FragmentNumber = start timestamp as string

Fragment_start_time % 10 == 0 and last 10 seconds

Query Range: 015-072
Kinesis returns 000-010, 010-020, ... , 070-080
Stored in s3:

	040-050 and
	080-090, 090-100, 100-110

Should be put to s3:

	010-020
	020-030
	030-040

	050-060
	060-070
	070-080
*/
func TestIngestIncient(t *testing.T) {
	ctx := context.Background()

	// map of keys in range to whether they are stored in s3
	allKeys := map[string]bool{
		"test-camera-uuid/1672600010.mkv": false,
		"test-camera-uuid/1672600020.mkv": false,
		"test-camera-uuid/1672600030.mkv": false,
		"test-camera-uuid/1672600040.mkv": true,
		"test-camera-uuid/1672600050.mkv": false,
		"test-camera-uuid/1672600060.mkv": false,
		"test-camera-uuid/1672600070.mkv": true,
	}

	var keysInS3 []string
	keysToPut := make(map[string]bool)
	for key, stored := range allKeys {
		if stored {
			keysInS3 = append(keysInS3, key)
		} else {
			// set to false first for all keys then validate that all keys are set to true
			keysToPut[key] = false
		}
	}

	test := testSetup{}
	test.s3.bucketName = "test-bucket"
	test.s3.storedKeys = keysInS3
	test.s3.keysThatReturnHeadError = []string{}
	test.kvs.dataEndpoint = "https://kinesisvideo.us-east-1.amazonaws.com"
	test.kvam.maxResultsPerListFragmentCall = 3

	test.query.cameraUUID = "test-camera-uuid"
	test.query.streamARN = "test-stream-arn"
	test.query.startTime = time.Unix(1672600015, 0).Add(100 * time.Millisecond)
	test.query.endTime = time.Unix(1672600072, 0).Add(400 * time.Millisecond)

	archiveBucketName := "test-bucket"
	preStartBufferDuration := time.Millisecond * 4200
	postEndBufferDuration := time.Millisecond * 2200

	inputIncident := incident.Incident{
		CameraUUID:           test.query.cameraUUID,
		StreamARN:            test.query.streamARN,
		StartFrameRelativeMs: int(test.query.startTime.Add(preStartBufferDuration).UnixMilli()),
		EndFrameRelativeMs:   int(test.query.endTime.Add(-postEndBufferDuration).UnixMilli()),
		PreStartBufferMs:     int(preStartBufferDuration.Milliseconds()),
		PostEndBufferMs:      int(postEndBufferDuration.Milliseconds()),
	}

	require.Equal(t, test.query.startTime, inputIncident.GetClipStartTime())
	require.Equal(t, test.query.endTime, inputIncident.GetClipEndTime())

	s3Client := test.newS3Client()
	kvsClient := test.newKVSClient()
	kvamClient := test.newKVAMClient()
	archiveClient := fragarchive.New(s3Client, archiveBucketName)

	ingestClient := ingest.Client{
		ArchiveClient: archiveClient,
		KVAMClient:    kvamClient,
		KVClient:      kvsClient,
	}

	err := ingestClient.HandleIncident(ctx, &inputIncident)
	require.NoError(t, err)

	assert.Equal(t, 2, kvsClient.GetDataEndpointCallCount())
	test.makeS3HeadObjectAssertions(t, s3Client)
	test.makeS3PutObjectAssertions(t, s3Client)
	test.makeKVAMGetMediaForFragmentListAssertions(t, kvamClient)
}
