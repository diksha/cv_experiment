// Package kvspusher provides a library for publishing fragments to kinesis video
package kvspusher

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo/types"
)

var emptyBodyChecksum = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

type fatalError struct {
	Err error
}

func (e fatalError) Error() string { return e.Err.Error() }
func (e fatalError) Unwrap() error { return e.Err }

// IsFatal returns true if the error reported by PutMedia is fatal
func IsFatal(err error) bool {
	var e *fatalError
	return errors.As(err, &e)
}

func fatalErrorf(format string, args ...interface{}) error {
	return fatalError{fmt.Errorf(format, args...)}
}

// PutMediaEvent is an event returned by the Kinesis Video PUT_MEDIA API
type PutMediaEvent struct {
	EventType string

	FragmentTimecode int64  `json:",omitempty"`
	FragmentNumber   string `json:",omitempty"`

	// trunk-ignore(golangci-lint/revive)
	ErrorId   int64  `json:",omitempty"`
	ErrorCode string `json:",omitempty"`
}

type putMediaError struct {
	PutMediaEvent
}

// Err returns an error value if this PutMediaEvent is an error
func (e PutMediaEvent) Err() error {
	if e.ErrorId == 0 {
		return nil
	}
	return putMediaError{e}
}

func (e putMediaError) Error() string {
	return fmt.Sprintf("PutMedia error: ErrorCode: %s, FragmentNumber: %s, FragmentTimecode: %d", e.ErrorCode, e.FragmentNumber, e.FragmentTimecode)
}

// KinesisVideoAPI specifies the Kinesis Video APIs required by the KvsPusher
// This is mostly used to mock the AWS APIs for tests.
type KinesisVideoAPI interface {
	GetDataEndpoint(context.Context, *kinesisvideo.GetDataEndpointInput, ...func(*kinesisvideo.Options)) (*kinesisvideo.GetDataEndpointOutput, error)
}

// HTTPClient can be used to override the default http client used by this client.
type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

// Client is a PutMedia client for Kinesis Video
type Client struct {
	streamName string
	awsConfig  aws.Config

	kvsAPI KinesisVideoAPI

	httpClient HTTPClient
	signer     v4.HTTPSigner
	endpoint   string
}

// ClientOpt is an option
type ClientOpt func(*Client)

// WithKinesisVideoAPI overrides the KinesisVideoAPI client which will be used for requests
func WithKinesisVideoAPI(api KinesisVideoAPI) ClientOpt {
	return func(c *Client) {
		c.kvsAPI = api
	}
}

// WithHTTPClient overrides the default http client which will be used for requests
func WithHTTPClient(client HTTPClient) ClientOpt {
	return func(c *Client) {
		c.httpClient = client
		c.awsConfig.HTTPClient = client
	}
}

// Init initialize a Kinesis Video PutMedia Client
func Init(ctx context.Context, streamName string, awsConfig aws.Config, opts ...ClientOpt) (*Client, error) {
	client := &Client{
		streamName: streamName,
		awsConfig:  awsConfig,
	}

	for _, optFn := range opts {
		optFn(client)
	}

	if client.kvsAPI == nil {
		client.kvsAPI = kinesisvideo.NewFromConfig(client.awsConfig)
	}

	if client.httpClient == nil {
		client.httpClient = http.DefaultClient
	}

	client.signer = v4.NewSigner()
	if err := client.refreshEndpoint(ctx); err != nil {
		return nil, fmt.Errorf("failed to get Kinesis Video PUT_MEDIA data endpoint: %w", err)
	}

	return client, nil
}

func (c *Client) refreshEndpoint(ctx context.Context) error {
	resp, err := c.kvsAPI.GetDataEndpoint(ctx, &kinesisvideo.GetDataEndpointInput{
		APIName:    types.APINamePutMedia,
		StreamName: aws.String(c.streamName),
	})
	if err != nil {
		return fmt.Errorf("failed to get PUT_MEDIA data endpoint from kvs: %w", err)
	}

	endpoint, err := url.Parse(aws.ToString(resp.DataEndpoint))
	if err != nil {
		return fmt.Errorf("failed to parse data endpoint url %q: %w", aws.ToString(resp.DataEndpoint), err)
	}

	endpoint.Path = "/putMedia"
	c.endpoint = endpoint.String()
	return nil
}

// PutMediaFragment takes a parsed fragment file and applies simplifications to it to reduce
// multiple clusters in the fragment into a single cluster and removes MKV elements which are unused
func (c *Client) PutMediaFragment(ctx context.Context, f *Fragment) error {

	var buf bytes.Buffer
	if err := WriteFragment(&buf, SimplifyFragment(f)); err != nil {
		// marshaling should never fail
		return fatalErrorf("failed to marshal mkv data")
	}

	return c.PutMedia(ctx, bytes.NewReader(buf.Bytes()), int64(buf.Len()))
}

// PutMedia calls the KinesisVideo PUT_MEDIA call against the data endpoint. This assumes that the data
// passed in is valid MKV data as required by the KVS PUT_MEDIA API
func (c *Client) PutMedia(ctx context.Context, body io.Reader, contentLength int64) error {
	req, err := http.NewRequestWithContext(ctx, "POST", c.endpoint, body)
	if err != nil {
		// request construction should never fail
		return fatalErrorf("failed to construct request: %w", err)
	}

	req.Header.Add("x-amzn-fragment-timecode-type", "ABSOLUTE")
	req.Header.Add("x-amzn-stream-name", c.streamName)
	req.ContentLength = contentLength

	creds, err := c.awsConfig.Credentials.Retrieve(ctx)
	if err != nil {
		// this error may not be fatal as it could be a temporary network failure
		// while attempting to refresh aws credentials
		return fmt.Errorf("failed to retrieve aws credentials: %w", err)
	}

	if err := c.signer.SignHTTP(ctx, creds, req, emptyBodyChecksum, "kinesisvideo", c.awsConfig.Region, time.Now()); err != nil {
		// signing should never fail
		return fatalErrorf("failed to sign request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != 200 {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("request failed with StatusCode=%d, failed to read body: %w", resp.StatusCode, err)
		}
		return fmt.Errorf("request failed with SatusCode=%d: %s", resp.StatusCode, string(body))
	}

	fragments := make(map[string]string)
	dec := json.NewDecoder(resp.Body)
	for dec.More() {
		var event PutMediaEvent
		if err := dec.Decode(&event); err != nil {
			return fmt.Errorf("failed to read response events: %w", err)
		}
		if event.Err() != nil {
			return fmt.Errorf("received error event: %w", event.Err())
		}
		if event.EventType == "IDLE" {
			// skip idle events
			continue
		}

		fragments[event.FragmentNumber] = event.EventType
	}

	if len(fragments) != 1 {
		return fmt.Errorf("request created %d fragments, expected to create 1 fragment", len(fragments))
	}

	return nil
}
