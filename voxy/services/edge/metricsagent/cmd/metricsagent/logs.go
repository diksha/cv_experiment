package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/voxel-ai/voxel/services/edge/metricsagent/lib/logpublisher"
	"github.com/voxel-ai/voxel/services/edge/metricsagent/lib/logtailer"
)

const logGroupTranscoder = "/voxel/edge/transcoder"
const logGroupGreengrass = "/voxel/edge/greengrass"

func createPublisher(ctx context.Context) *logpublisher.Client {
	publisher, err := logpublisher.New(
		ctx,
		logpublisher.Config{
			BatchSize:             1000,
			MaxPublishingInterval: time.Minute,
			StreamTrackRetention:  time.Hour,
		},
	)

	if err != nil {
		log.Fatalf("failed to create log publisher: %v", err)
	}

	return publisher
}

func watchFiles(logGroupToFileGlobs map[string][]string) <-chan logpublisher.LogAndDestination {
	aggregateChannel := make(chan logpublisher.LogAndDestination)

	if len(logGroupToFileGlobs) == 0 {
		log.Fatal("failed to watch log files, no globs provided")
	}

	for logGroup, globs := range logGroupToFileGlobs {
		watcher, err := logtailer.Watch(globs...)
		if err != nil {
			log.Fatalf("failed to setup watcher on globs %v: %v", globs, err)
		}

		go func(logWatcher *logtailer.Watcher, logGroup string) {
			for v := range logWatcher.Lines() {
				aggregateChannel <- logpublisher.LogAndDestination{
					Log: logpublisher.Log{
						Message:   v.Text,
						Timestamp: time.Now(),
					},
					Stream: logpublisher.Destination{
						LogGroup:  logGroup,
						LogStream: fmt.Sprintf("%v - %v", *edgeUUID, v.Filename),
					},
				}
			}
		}(watcher, logGroup)
	}

	return aggregateChannel
}

func publishLogFiles(ctx context.Context) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	publisher := createPublisher(ctx)

	logGroupToFileGlobs := map[string][]string{
		logGroupTranscoder: {
			"/var/log/containers/edge-transcoder-*.log",
		},
		logGroupGreengrass: {
			"/greengrass/v2/logs/voxel.edge.QuicksyncTranscoder.log",
		},
	}

	for logLine := range watchFiles(logGroupToFileGlobs) {
		err := publisher.Push(ctx, logLine.Log, logLine.Stream)
		if err != nil {
			log.Fatalf("failed to push log to publisher: %v", err)
		}
	}
}
