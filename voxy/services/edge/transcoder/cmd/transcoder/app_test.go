package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg/ffmpegbazel"
	"github.com/voxel-ai/voxel/go/edge/metricsctx"
	"github.com/voxel-ai/voxel/lib/utils/aws/fake"
	edgeconfigpb "github.com/voxel-ai/voxel/protos/edge/edgeconfig/v1"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/kvspusher"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/transcoder"
)

func init() {
	if err := ffmpegbazel.Find(); err != nil {
		panic(err)
	}
}

func logCtx(ctx context.Context, t *testing.T) context.Context {
	return zerolog.New(zerolog.NewTestWriter(t)).WithContext(ctx)
}

func mockMetricsCtx(ctx context.Context) (context.Context, *fake.CloudwatchClient) {
	fake := &fake.CloudwatchClient{}
	ctx = metricsctx.WithPublisher(ctx, metricsctx.Config{
		Client:    fake,
		Namespace: "voxel/testing",
		Interval:  100 * time.Millisecond,
	})
	return ctx, fake
}

func TestMainApp(t *testing.T) {
	ctx := logCtx(context.Background(), t)
	ctx, cancel := context.WithTimeout(ctx, 1*time.Minute)
	defer cancel()

	mediaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := json.Marshal(kvspusher.PutMediaEvent{
			FragmentTimecode: 1,
			FragmentNumber:   "1",
		})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Add("Content-Type", "application/json")
		// trunk-ignore(semgrep/go.lang.security.audit.xss.no-direct-write-to-responsewriter.no-direct-write-to-responsewriter)
		_, err = w.Write(body)
		require.NoError(t, err, "must write response body")
	}))
	defer mediaSrv.Close()

	ctx, cwapi := mockMetricsCtx(ctx)
	kvsapi := &fake.KinesisVideoClient{}
	kvsapi.GetDataEndpointReturns(&kinesisvideo.GetDataEndpointOutput{
		DataEndpoint: &mediaSrv.URL,
	}, nil)

	app := &App{
		Config: Config{
			FFmpeg: ConfigFFmpeg{
				LogLevel: "debug",
			},
		},
		Debug: AppDebug{
			KinesisVideoAPI: kvsapi,
			ForceSoftware:   true,
			Input:           transcoder.TestInput(480, 360, 5, false, 5*time.Second),
			StreamConfig:    &edgeconfigpb.StreamConfig{},
			AWSConfig: &aws.Config{
				Credentials: credentials.NewStaticCredentialsProvider("some-key", "some-secret", "some-session"),
			},
		},
	}

	assert.NoError(t, app.Run(ctx), "app should run without erroring")
	assert.GreaterOrEqual(t, 1, len(cwapi.Invocations()), "should have called cloudwatch at least once")
	assert.GreaterOrEqual(t, 1, kvsapi.GetDataEndpointCallCount(), "should have called GetDataEndpoint at least once")
}
