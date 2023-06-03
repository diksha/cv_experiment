package logpublisher

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"

	cwlogs "github.com/aws/aws-sdk-go-v2/service/cloudwatchlogs"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatchlogs/types"
)

// Log is a combination of a message and timestamp
type Log struct {
	Message   string
	Timestamp time.Time
}

// Destination specifies the log group and stream name in CloudWatchLogs
type Destination struct {
	LogGroup  string
	LogStream string
}

// LogAndDestination combines log with a stream it is destined for
type LogAndDestination struct {
	Log    Log
	Stream Destination
}

func getCloudwatchLogsClient(ctx context.Context) (*cwlogs.Client, error) {
	loadCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	config, err := awsconfig.LoadDefaultConfig(loadCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to load default aws config: %w", err)
	}

	return cwlogs.NewFromConfig(config), nil
}

// CWLogsClient is an interface for creating streams and putting logs to cloudwatch logs
type CWLogsClient interface {
	CreateLogStream(ctx context.Context, params *cwlogs.CreateLogStreamInput, optFns ...func(*cwlogs.Options)) (*cwlogs.CreateLogStreamOutput, error)
	PutLogEvents(ctx context.Context, params *cwlogs.PutLogEventsInput, optFns ...func(*cwlogs.Options)) (*cwlogs.PutLogEventsOutput, error)
}

// Config allows you to set optional parameters for
type Config struct {
	// maximum number of logs to buffer before publishing
	BatchSize int
	// interval at which log events will be published
	MaxPublishingInterval time.Duration
	// longest time between pushes to a stream before we no longer track the channel
	StreamTrackRetention time.Duration
}

// Client provides an interface for publishing logs to CloudWatch logs
type Client struct {
	cloudwatchClient CWLogsClient
	channels         map[Destination]chan types.InputLogEvent
	latestLogTime    map[Destination]time.Time
	config           Config
}

// New creates a publishing client
func New(ctx context.Context, config Config) (*Client, error) {
	return NewWithCWLClient(ctx, config, nil)
}

// NewWithCWLClient creates a publishing client
// cloudWatchLogs is optional, set to nil to fetch using default AWS configs
func NewWithCWLClient(ctx context.Context, config Config, cloudWatchLogs CWLogsClient) (*Client, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var client CWLogsClient
	var err error
	if cloudWatchLogs == nil {
		client, err = getCloudwatchLogsClient(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to create log publisher: %w", err)
		}
	} else {
		client = cloudWatchLogs
	}

	if config.BatchSize <= 0 {
		return nil, errors.New("batch size must be >= 0")
	}

	if config.MaxPublishingInterval <= 0 {
		return nil, errors.New("max publishing interval must be >= 0")
	}

	if config.StreamTrackRetention <= 0 {
		return nil, errors.New("stream track retention must be > 0")
	}

	publisher := Client{
		cloudwatchClient: client,
		channels:         make(map[Destination]chan types.InputLogEvent),
		latestLogTime:    make(map[Destination]time.Time),
		config:           config,
	}

	return &publisher, nil
}

// Push queues a log message to be published to the given stream
// Must only be called from a single goroutine
func (p *Client) Push(ctx context.Context, log Log, stream Destination) error {
	p.closeStaleChannels()

	timeMilli := log.Timestamp.UnixMilli()
	event := types.InputLogEvent{
		Message:   &log.Message,
		Timestamp: &timeMilli,
	}

	// Ensure timestamp ordering
	latest, ok := p.latestLogTime[stream]
	if ok && log.Timestamp.Before(latest) {
		return fmt.Errorf(
			"Cannot push out of order log with timestamp %v to stream (%q, %q) since latest log has timestamp %v",
			log.Timestamp.UnixMilli(),
			stream.LogGroup,
			stream.LogStream,
			latest.UnixMilli(),
		)
	}
	p.latestLogTime[stream] = log.Timestamp

	// Ensure stream exists
	_, ok = p.channels[stream]
	if !ok {
		if err := p.createLogStream(ctx, stream); err != nil {
			return fmt.Errorf("failed to create stream for log: %w", err)
		}

		p.channels[stream] = make(chan types.InputLogEvent)
		go p.work(ctx, stream, p.channels[stream])
	}

	// Push event to worker
	p.channels[stream] <- event

	return nil
}

// Close indicates that no more logs will be pushed to the publisher
func (p *Client) Close() {
	for dest := range p.channels {
		p.closeChannel(dest)
	}
}

func (p *Client) closeChannel(dest Destination) {
	close(p.channels[dest])
	delete(p.channels, dest)
	delete(p.latestLogTime, dest)
}

func (p *Client) closeStaleChannels() {
	staleTime := time.Now().Add(-p.config.StreamTrackRetention)
	for dest, timestamp := range p.latestLogTime {
		if timestamp.Before(staleTime) {
			p.closeChannel(dest)
		}
	}
}

func (p *Client) work(ctx context.Context, stream Destination, eventCh chan types.InputLogEvent) {
	events := []types.InputLogEvent{}
	ticker := time.NewTicker(p.config.MaxPublishingInterval)

	var event types.InputLogEvent
	channelOpen := true
	for {
		performPublish := false
		select {
		case <-ticker.C:
			performPublish = true
		case event, channelOpen = <-eventCh:
			if channelOpen {
				events = append(events, event)
				performPublish = (len(events) == p.config.BatchSize)
			} else {
				performPublish = true
			}
		}

		if performPublish {
			if err := p.publish(ctx, stream, events); err != nil {
				log.Printf("failed to publish logs: %v", err)
			}
			events = nil
		}

		if !channelOpen {
			break
		}
	}
}

func (p *Client) createLogStream(ctx context.Context, dest Destination) error {
	_, err := p.cloudwatchClient.CreateLogStream(ctx, &cwlogs.CreateLogStreamInput{
		LogGroupName:  &dest.LogGroup,
		LogStreamName: &dest.LogStream,
	})

	if err != nil {
		var alreadyExistsErr *types.ResourceAlreadyExistsException
		if errors.As(err, &alreadyExistsErr) {
			return nil // suppress this error since we only care that the stream exists
		}

		return fmt.Errorf("cloudwatchlogs CreateLogStream call failed for log stream (%v, %v): %w", dest.LogGroup, dest.LogStream, err)
	}

	return nil
}

func (p *Client) publish(ctx context.Context, dest Destination, events []types.InputLogEvent) error {
	if len(events) == 0 {
		return nil
	}

	input := cwlogs.PutLogEventsInput{
		LogEvents:     events,
		LogGroupName:  &dest.LogGroup,
		LogStreamName: &dest.LogStream,
	}

	_, err := p.cloudwatchClient.PutLogEvents(ctx, &input)
	if err != nil {
		err = fmt.Errorf(
			"cloudwatchlogs PutLogEvents call failed for log group %q, stream %q. No retries - clearing logs: %w",
			dest.LogGroup,
			dest.LogStream,
			err,
		)
	}

	return err
}
