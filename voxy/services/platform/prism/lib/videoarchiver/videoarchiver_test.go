package videoarchiver_test

import (
	"context"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo"
	"github.com/stretchr/testify/assert"

	"github.com/voxel-ai/voxel/lib/utils/aws/fake"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/fragarchive"
	"github.com/voxel-ai/voxel/services/platform/prism/lib/videoarchiver"
)

func stubGetDataEndpoint(fakeKVClient *fake.KinesisVideoClient) {
	fakeKVClient.GetDataEndpointStub = func(context.Context, *kinesisvideo.GetDataEndpointInput, ...func(*kinesisvideo.Options)) (*kinesisvideo.GetDataEndpointOutput, error) {
		dataEndpoint := "https://kinesisvideo.us-east-1.amazonaws.com"
		return &kinesisvideo.GetDataEndpointOutput{
			DataEndpoint: &dataEndpoint,
		}, nil
	}
}

func TestNew(t *testing.T) {
	ctx := context.Background()
	archive := fragarchive.New(new(fake.S3Client), "test-bucket")
	kvClient := new(fake.KinesisVideoClient)
	kvamClient := new(fake.KinesisVideoArchivedMediaClient)

	stubGetDataEndpoint(kvClient)

	videoarchiveClient, err := videoarchiver.New(ctx, videoarchiver.Config{
		FragmentArchiveClient:           archive,
		KinesisVideoStreamARN:           "test-stream-arn",
		KinesisVideoClient:              kvClient,
		KinesisVideoArchivedMediaClient: kvamClient,
		CameraUUID:                      "test-camera-uuid",
	})

	assert.NotNil(t, videoarchiveClient)
	assert.NoError(t, err)
}
