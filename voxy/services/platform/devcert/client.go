package devcert

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/restjson"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/config"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/encoding/protojson"

	devcertpb "github.com/voxel-ai/voxel/protos/platform/devcert/v1"
)

const (
	developerCertUrl = "https://x7gq3v6ftxrplwxywt4c3bo5fq0sycfh.lambda-url.us-west-2.on.aws/"
)

// We want it to be possible to call this as a grpc endpoint in the future
// so we ensure that this satisfies a grpc service definition
var _ devcertpb.DevCertServiceClient = (*Client)(nil)

type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

type Client struct {
	Endpoint    string
	HTTPClient  HTTPClient
	Credentials aws.CredentialsProvider
}

func (c *Client) GetDevCert(ctx context.Context, req *devcertpb.GetDevCertRequest, opts ...grpc.CallOption) (*devcertpb.GetDevCertResponse, error) {
	if len(opts) > 0 {
		return nil, fmt.Errorf("no options are currently supported in this client")
	}

	// marshal the request to json
	reqBody, err := protojson.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// get the endpoint, using the default endpoint if unset
	endpoint := c.Endpoint
	if endpoint == "" {
		endpoint = developerCertUrl
	}

	// construct the new http request
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// fetch aws credentials, defaulting to default creds if unset
	credsProvider := c.Credentials
	if credsProvider == nil {
		cfg, err := config.LoadDefaultConfig(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to load default AWS config: %w", err)
		}
		credsProvider = cfg.Credentials
	}

	creds, err := credsProvider.Retrieve(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get AWS credentials: %w", err)
	}

	// compute the payload hash
	h := sha256.New()
	_, err = h.Write(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to compute payload hash: %w", err)
	}
	payloadHash := hex.EncodeToString(h.Sum(nil))

	// sign the request with aws sigv4 signing
	signer := v4.NewSigner()
	httpReq.ContentLength = int64(len(reqBody))
	httpReq.Header.Add("Content-Type", "application/json")
	err = signer.SignHTTP(ctx, creds, httpReq, payloadHash, "lambda", "us-west-2", time.Now())
	if err != nil {
		return nil, fmt.Errorf("failed to sign lambda http request: %w", err)
	}

	// perform the request with the passed in client, or use the default client if unset
	httpClient := c.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failure: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	// read the body
	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// handle errors
	if httpResp.StatusCode != http.StatusOK {
		errorType, msg, err := restjson.GetErrorInfo(json.NewDecoder(bytes.NewReader(respBody)))
		if err != nil {
			return nil, fmt.Errorf("got unexpected response code %d, failed to read error info: %w", httpResp.StatusCode, err)
		}

		return nil, fmt.Errorf("error response from api %v: %v", errorType, msg)
	}

	// unmarshal response on success
	resp := &devcertpb.GetDevCertResponse{}
	err = protojson.Unmarshal(respBody, resp)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return resp, nil
}

type Cert struct {
	RootCA []byte
	Cert   []byte
	Key    []byte
}
