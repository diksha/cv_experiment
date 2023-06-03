package devcert_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/voxel-ai/voxel/lib/utils/gofakes"
	devcertpb "github.com/voxel-ai/voxel/protos/platform/devcert/v1"
	"github.com/voxel-ai/voxel/services/platform/devcert"
)

func TestClientError(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	httpclient := &gofakes.FakeHTTPClient{}
	client := &devcert.Client{
		Endpoint:    "https://test.endpoint.local/",
		HTTPClient:  httpclient,
		Credentials: credentials.NewStaticCredentialsProvider("", "", ""),
	}

	httpclient.DoReturns(nil, fmt.Errorf("api failure"))
	_, err := client.GetDevCert(ctx, &devcertpb.GetDevCertRequest{})
	require.Error(t, err, "get dev cert should error")
}

func TestClientSuccess(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	httpclient := &gofakes.FakeHTTPClient{}
	client := &devcert.Client{
		Endpoint:    "https://test.endpoint.local/",
		HTTPClient:  httpclient,
		Credentials: credentials.NewStaticCredentialsProvider("fake", "creds", ""),
	}

	csrdata := []byte("fake csr test data")
	certdata := []byte("fake cert test data")

	// write a stub for the response
	httpclient.DoStub = func(req *http.Request) (*http.Response, error) {
		respBody, err := protojson.Marshal(&devcertpb.GetDevCertResponse{
			Cert: string(certdata),
		})
		require.NoError(t, err, "should marshal test request body")

		resp := &http.Response{
			Body:       io.NopCloser(bytes.NewReader(respBody)),
			StatusCode: http.StatusOK,
		}
		return resp, nil
	}

	res, err := client.GetDevCert(ctx, &devcertpb.GetDevCertRequest{
		Csr: string(csrdata),
	})
	require.NoError(t, err, "get dev cert should not error")
	require.NotNil(t, res, "result should not be nil")
	assert.Equal(t, certdata, []byte(res.Cert), "result cert data should be correct")
}
