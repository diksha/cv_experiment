package main

import (
	"context"
	"errors"
	"sync"

	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo"
	kinesisvideotypes "github.com/aws/aws-sdk-go-v2/service/kinesisvideo/types"
)

var globalEndpointCacheManager = &endpointCacheManager{
	endpointCache: make(map[endpointIdentifier]string),
}

type endpointIdentifier struct {
	APIName  kinesisvideotypes.APIName
	StreamID string
}

type endpointCacheManager struct {
	endpointCache map[endpointIdentifier]string
	sync.RWMutex
}

func (c *endpointCacheManager) GetEndpoint(endpointID endpointIdentifier) (string, bool) {
	c.RLock()
	defer c.RUnlock()

	endpoint, ok := c.endpointCache[endpointID]
	return endpoint, ok
}

func (c *endpointCacheManager) SetEndpoint(endpointID endpointIdentifier, endpoint string) {
	c.Lock()
	defer c.Unlock()

	c.endpointCache[endpointID] = endpoint
}

type kvClientWithEndpointCaching struct {
	innerClient   *kinesisvideo.Client
	endpointCache *endpointCacheManager
}

func (client kvClientWithEndpointCaching) GetDataEndpoint(ctx context.Context, params *kinesisvideo.GetDataEndpointInput, optFns ...func(*kinesisvideo.Options)) (*kinesisvideo.GetDataEndpointOutput, error) {
	var streamID string
	if params.StreamARN != nil {
		streamID = *params.StreamARN
	} else if params.StreamName != nil {
		streamID = *params.StreamName
	} else {
		return nil, errors.New("must specify exactly one of StreamName or StreamARN")
	}

	endpointID := endpointIdentifier{
		APIName:  params.APIName,
		StreamID: streamID,
	}

	endpoint, ok := client.endpointCache.GetEndpoint(endpointID)

	if ok {
		return &kinesisvideo.GetDataEndpointOutput{
			DataEndpoint: &endpoint,
		}, nil
	}

	output, err := client.innerClient.GetDataEndpoint(ctx, params, optFns...)
	if err == nil {
		client.endpointCache.SetEndpoint(endpointID, *output.DataEndpoint)
	}

	// trunk-ignore(golangci-lint/wrapcheck)
	return output, err
}
