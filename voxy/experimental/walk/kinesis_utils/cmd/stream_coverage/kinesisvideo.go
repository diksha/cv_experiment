package main

import (
	"context"
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo"
	kinesisvideotypes "github.com/aws/aws-sdk-go-v2/service/kinesisvideo/types"
	kvam "github.com/aws/aws-sdk-go-v2/service/kinesisvideoarchivedmedia"
	kvamtypes "github.com/aws/aws-sdk-go-v2/service/kinesisvideoarchivedmedia/types"
)

// KinesisVideoAPI is the set of API functions required by this library to interact with from Kinesis Video
type KinesisVideoAPI interface {
	GetDataEndpoint(ctx context.Context, params *kinesisvideo.GetDataEndpointInput, optFns ...func(*kinesisvideo.Options)) (*kinesisvideo.GetDataEndpointOutput, error)
}

var _ KinesisVideoAPI = (*kinesisvideo.Client)(nil)

// KinesisVideoArchivedMediaAPI is the set of API functions required by this library to interact with from Kinesis Video Archived Media
type KinesisVideoArchivedMediaAPI interface {
	ListFragments(ctx context.Context, params *kvam.ListFragmentsInput, optFns ...func(*kvam.Options)) (*kvam.ListFragmentsOutput, error)
}

var _ KinesisVideoArchivedMediaAPI = (*kvam.Client)(nil)

type streamIdentifier struct {
	streamName string
	streamARN  string
}

func streamIdentiferFromARN(streamARN string) streamIdentifier {
	return streamIdentifier{
		streamARN: streamARN,
	}
}

func (id *streamIdentifier) ARN() *string {
	if id.streamARN == "" {
		return nil
	}

	return &id.streamARN
}

func (id *streamIdentifier) Name() *string {
	if id.streamName == "" {
		return nil
	}
	return &id.streamName
}

type kvsClient struct {
	StreamID                        streamIdentifier
	KinesisVideoClient              KinesisVideoAPI
	KinesisVideoArchivedMediaClient KinesisVideoArchivedMediaAPI
	EndpointURLS                    map[kinesisvideotypes.APIName]string
}

func newKVSClient(
	ctx context.Context,
	streamID streamIdentifier,
	kinesisVideoClient KinesisVideoAPI,
	kinesisVideoArchivedMedia KinesisVideoArchivedMediaAPI,
) (*kvsClient, error) {
	client := &kvsClient{
		StreamID:                        streamID,
		KinesisVideoClient:              kinesisVideoClient,
		KinesisVideoArchivedMediaClient: kinesisVideoArchivedMedia,
	}

	// populate data endpoints initially for thread safety
	requiredAPIEndpoints := [...]kinesisvideotypes.APIName{
		kinesisvideotypes.APINameListFragments,
	}

	client.EndpointURLS = make(map[kinesisvideotypes.APIName]string)
	for _, apiName := range requiredAPIEndpoints {
		err := client.fetchDataEndpointURL(ctx, apiName)
		if err != nil {
			return nil, err
		}
	}
	return client, nil
}

func (client *kvsClient) fetchDataEndpointURL(ctx context.Context, apiName kinesisvideotypes.APIName) error {
	endpoint, err := client.KinesisVideoClient.GetDataEndpoint(
		ctx,
		&kinesisvideo.GetDataEndpointInput{
			APIName:    apiName,
			StreamARN:  client.StreamID.ARN(),
			StreamName: client.StreamID.Name(),
		},
	)
	if err != nil {
		return fmt.Errorf("failed call to GetDataEndpoint: %w", err)
	}

	client.EndpointURLS[apiName] = *endpoint.DataEndpoint
	return nil
}

func (client *kvsClient) ListFragments(ctx context.Context, start, end time.Time) ([]kvamtypes.Fragment, error) {
	endpoint, ok := client.EndpointURLS[kinesisvideotypes.APINameListFragments]
	if !ok {
		return nil, fmt.Errorf("Did not initialize data endpoint URL for %q", kinesisvideotypes.APINameListFragments)
	}

	endpointResolver := kvam.EndpointResolverFromURL(endpoint)
	input := &kvam.ListFragmentsInput{
		FragmentSelector: &kvamtypes.FragmentSelector{
			FragmentSelectorType: kvamtypes.FragmentSelectorTypeProducerTimestamp,
			TimestampRange: &kvamtypes.TimestampRange{
				StartTimestamp: &start,
				EndTimestamp:   &end,
			},
		},
		NextToken:  nil,
		StreamARN:  client.StreamID.ARN(),
		StreamName: client.StreamID.Name(),
	}

	fragments := []kvamtypes.Fragment{}
	for {
		result, err := client.
			KinesisVideoArchivedMediaClient.
			ListFragments(
				ctx,
				input,
				kvam.WithEndpointResolver(endpointResolver),
			)
		if err != nil {
			return nil, fmt.Errorf("failed call to ListFragments: %w", err)
		}

		fragments = append(fragments, result.Fragments...)

		if result.NextToken == nil {
			break
		}

		input.NextToken = result.NextToken
	}

	return fragments, nil
}
