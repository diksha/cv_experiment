package main

import (
	"context"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/kinesisvideo"
)

func getKinesisVideoStreams(ctx context.Context, awsConfig aws.Config) ([]string, error) {
	kvs := kinesisvideo.NewFromConfig(awsConfig)
	var nextToken *string
	var streams []string

	for {
		resp, err := kvs.ListStreams(ctx, &kinesisvideo.ListStreamsInput{
			NextToken: nextToken,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to list kinesis video streams: %w", err)
		}
		for _, stream := range resp.StreamInfoList {
			streams = append(streams, aws.ToString(stream.StreamName))
		}
		if resp.NextToken == nil {
			break
		}

		nextToken = resp.NextToken

	}
	return streams, nil
}

func getTagsForStream(ctx context.Context, awsConfig aws.Config, streamName string) (map[string]string, error) {
	kvs := kinesisvideo.NewFromConfig(awsConfig)
	resp, err := kvs.ListTagsForStream(ctx, &kinesisvideo.ListTagsForStreamInput{
		StreamName: aws.String(streamName),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list tags for %q: %w", streamName, err)
	}

	return resp.Tags, nil
}

func getEdgeForStream(ctx context.Context, awsConfig aws.Config, streamName string) (string, error) {
	tags, err := getTagsForStream(ctx, awsConfig, streamName)
	if err != nil {
		return "", fmt.Errorf("tag request failed: %w", err)
	}

	if edge, ok := tags["edge:allowed-uuid:primary"]; ok {
		return edge, nil
	}

	return "", fmt.Errorf("no edge found for stream %q", streamName)
}

func setTagsForStream(ctx context.Context, awsConfig aws.Config, streamName string, tags map[string]string) error {
	kvs := kinesisvideo.NewFromConfig(awsConfig)
	_, err := kvs.TagStream(ctx, &kinesisvideo.TagStreamInput{
		StreamName: aws.String(streamName),
		Tags:       tags,
	})
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	return nil
}
