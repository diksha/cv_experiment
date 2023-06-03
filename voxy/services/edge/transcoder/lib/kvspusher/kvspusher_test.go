package kvspusher_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/lib/utils/aws/fake"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/kvspusher"
)

func TestPutMedia(t *testing.T) {
	testFrag := &kvspusher.Fragment{
		EBML: kvspusher.DefaultMKVHeader,
		Segment: kvspusher.Segment{
			Info: kvspusher.Info{
				Title: "test-fragment",
			},
		},
	}

	// set up a mock putMedia handler
	putMediaHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		frag, err := kvspusher.ReadFragment(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to read input fragment: %v", err), http.StatusBadRequest)
		}

		assert.EqualValues(t, 1, len(frag.Segment.Cluster), "fragment sent to putMedia has 1 cluster")

		event := kvspusher.PutMediaEvent{
			EventType:        "PERSISTED",
			FragmentTimecode: 1,
			FragmentNumber:   "1",
		}
		body, err := json.Marshal(event)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to marshal json response: %v", err), http.StatusInternalServerError)
			return
		}

		if _, err = w.Write(body); err != nil {
			t.Logf("failed to write response data: %v", err)
		}
	})
	srv := httptest.NewServer(putMediaHandler)
	defer srv.Close()

	kvsapi := new(fake.KinesisVideoClient)
	kvsapi.GetDataEndpointReturns(&kinesisvideo.GetDataEndpointOutput{
		DataEndpoint: aws.String(srv.URL),
	}, nil)
	awsConfig := aws.Config{
		Credentials: credentials.NewStaticCredentialsProvider("test-key", "test-secret", ""),
	}
	pusher, err := kvspusher.Init(context.Background(), "test-stream", awsConfig, kvspusher.WithKinesisVideoAPI(kvsapi))
	require.NoError(t, err, "pusher initializes")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	err = pusher.PutMediaFragment(ctx, testFrag)
	require.NoError(t, err, "pusher does not error")

	assert.EqualValues(t, 1, kvsapi.GetDataEndpointCallCount(), "GetDataEndpoint is called once")
}
