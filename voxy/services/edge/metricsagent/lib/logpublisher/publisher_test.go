package logpublisher_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	cwlogs "github.com/aws/aws-sdk-go-v2/service/cloudwatchlogs"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	. "github.com/voxel-ai/voxel/services/edge/metricsagent/lib/logpublisher"
)

type mockCloudWatchLogsClient struct {
	createStreamChan chan Destination
	putLogsChan      chan *cwlogs.PutLogEventsInput
}

func (client mockCloudWatchLogsClient) CreateLogStream(ctx context.Context, params *cwlogs.CreateLogStreamInput, optFns ...func(*cwlogs.Options)) (*cwlogs.CreateLogStreamOutput, error) {
	client.createStreamChan <- Destination{
		LogGroup:  *params.LogGroupName,
		LogStream: *params.LogStreamName,
	}
	return nil, nil
}

func (client mockCloudWatchLogsClient) PutLogEvents(ctx context.Context, params *cwlogs.PutLogEventsInput, optFns ...func(*cwlogs.Options)) (*cwlogs.PutLogEventsOutput, error) {
	client.putLogsChan <- params
	return nil, nil
}

func TestConfig(t *testing.T) {
	ctx := context.Background()

	client := mockCloudWatchLogsClient{
		createStreamChan: make(chan Destination),
		putLogsChan:      make(chan *cwlogs.PutLogEventsInput),
	}

	var err error
	_, err = NewWithCWLClient(ctx, Config{
		MaxPublishingInterval: time.Millisecond * 10,
		BatchSize:             1,
		StreamTrackRetention:  time.Hour,
	}, client)
	assert.NoError(t, err, "should not error with good config")

	// Zero Values
	_, err = NewWithCWLClient(ctx, Config{
		BatchSize:            1,
		StreamTrackRetention: time.Hour,
	}, client)
	assert.Error(t, err, "should error with MaxPublishingInterval = 0")
	_, err = NewWithCWLClient(ctx, Config{
		MaxPublishingInterval: time.Millisecond * 10,
		StreamTrackRetention:  time.Hour,
	}, client)
	assert.Error(t, err, "should error with BatchSize = 0")
	_, err = NewWithCWLClient(ctx, Config{
		MaxPublishingInterval: time.Millisecond * 10,
		BatchSize:             1,
	}, client)
	assert.Error(t, err, "should error with StreamTrackRetention = 0")

	// Negative Values
	_, err = NewWithCWLClient(ctx, Config{
		MaxPublishingInterval: -time.Millisecond * 10,
		BatchSize:             1,
		StreamTrackRetention:  time.Hour,
	}, client)
	assert.Error(t, err, "should error with negative MaxPublishingInterval")
	_, err = NewWithCWLClient(ctx, Config{
		MaxPublishingInterval: time.Millisecond * 10,
		BatchSize:             -1,
		StreamTrackRetention:  time.Hour,
	}, client)
	assert.Error(t, err, "should error with negative BatchSize")
	_, err = NewWithCWLClient(ctx, Config{
		MaxPublishingInterval: time.Millisecond * 10,
		BatchSize:             1,
		StreamTrackRetention:  -time.Hour,
	}, client)
	assert.Error(t, err, "should error with negative StreamTrackRetention")
}

func TestOutOfOrderPublish(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	client := mockCloudWatchLogsClient{
		createStreamChan: make(chan Destination),
		putLogsChan:      make(chan *cwlogs.PutLogEventsInput),
	}

	config := Config{
		MaxPublishingInterval: time.Millisecond * 10,
		BatchSize:             1,
		StreamTrackRetention:  time.Hour,
	}
	publisher, err := NewWithCWLClient(ctx, config, client)
	require.NoError(t, err, "creating publisher should not error")
	defer publisher.Close()

	dest := Destination{
		LogGroup:  "group",
		LogStream: "stream",
	}
	now := time.Now()

	testDone := make(chan struct{})
	go func() {
		<-client.createStreamChan
		<-client.putLogsChan

		// this should block since we dont want the out of order log to publish
		<-client.putLogsChan
		close(testDone)
	}()

	err = publisher.Push(
		ctx,
		Log{
			Message:   "message",
			Timestamp: now,
		},
		dest,
	)
	assert.NoError(t, err, "should not error on first push")

	err = publisher.Push(
		ctx,
		Log{
			Message:   "message",
			Timestamp: now.Add(-time.Millisecond),
		},
		dest,
	)
	assert.Error(t, err, "should error on out of order log push")

	select {
	case <-testDone:
		assert.Fail(t, "should not publish out of order log")
	case <-ctx.Done():
	}
}

func TestPublishOnBatchSize(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	client := mockCloudWatchLogsClient{
		createStreamChan: make(chan Destination),
		putLogsChan:      make(chan *cwlogs.PutLogEventsInput),
	}

	config := Config{
		MaxPublishingInterval: time.Hour,
		BatchSize:             2,
		StreamTrackRetention:  time.Hour,
	}
	publisher, err := NewWithCWLClient(ctx, config, client)
	require.NoError(t, err, "creating publisher should not error")
	defer publisher.Close()

	dest := Destination{
		LogGroup:  "group",
		LogStream: "stream",
	}

	testDone := make(chan struct{})
	go func() {
		assert.Equal(t, dest, <-client.createStreamChan, "should create stream")

		for i := 0; i < 5; i++ {
			logs := <-client.putLogsChan
			assert.Len(t, logs.LogEvents, 2, "should output in correct batch size")
		}

		close(testDone)
	}()

	for i := 0; i < 10; i++ {
		log := Log{
			Message:   "message",
			Timestamp: time.Now(),
		}
		assert.NoError(t, publisher.Push(ctx, log, dest))
	}

	select {
	case <-ctx.Done():
		assert.Fail(t, "test timed out")
	case <-testDone:
	}
}

func TestPublishOnInterval(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	client := mockCloudWatchLogsClient{
		createStreamChan: make(chan Destination),
		putLogsChan:      make(chan *cwlogs.PutLogEventsInput),
	}

	config := Config{
		MaxPublishingInterval: 100 * time.Millisecond,
		BatchSize:             1000,
		StreamTrackRetention:  time.Hour,
	}
	publisher, err := NewWithCWLClient(ctx, config, client)
	require.NoError(t, err, "creating publisher should not error")
	defer publisher.Close()

	dest := Destination{
		LogGroup:  "group",
		LogStream: "stream",
	}

	testDone := make(chan struct{})
	go func() {
		// Create Stream
		assert.Equal(t, dest, <-client.createStreamChan, "should create stream")

		logs := <-client.putLogsChan
		assert.Len(t, logs.LogEvents, 5, "should publish after first 5 logs")
		logs = <-client.putLogsChan
		assert.Len(t, logs.LogEvents, 1, "should publish last log separately")

		close(testDone)
	}()

	for i := 0; i < 5; i++ {
		log := Log{
			Message:   "message",
			Timestamp: time.Now(),
		}
		assert.NoError(t, publisher.Push(ctx, log, dest))
	}

	time.Sleep(100 * time.Millisecond)

	log := Log{
		Message:   "message",
		Timestamp: time.Now(),
	}

	assert.NoError(t, publisher.Push(ctx, log, dest))

	select {
	case <-ctx.Done():
		assert.Fail(t, "test timed out")
	case <-testDone:
	}
}

func TestCloseStaleChannelsOnPush(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	client := mockCloudWatchLogsClient{
		createStreamChan: make(chan Destination),
		putLogsChan:      make(chan *cwlogs.PutLogEventsInput),
	}

	config := Config{
		MaxPublishingInterval: 10 * time.Millisecond,
		BatchSize:             1,
		StreamTrackRetention:  time.Millisecond,
	}
	publisher, err := NewWithCWLClient(ctx, config, client)
	require.NoError(t, err, "creating publisher should not error")
	defer publisher.Close()

	dest := Destination{
		LogGroup:  "group",
		LogStream: "stream",
	}

	staleTimestamp := time.Now().Add(-1 * time.Minute) // stale compared to stream retention

	testDone := make(chan struct{})
	go func() {
		for i := 0; i < 2; i++ {
			log := Log{
				Message:   fmt.Sprintf("message%v", i),
				Timestamp: staleTimestamp,
			}
			assert.NoError(t, publisher.Push(ctx, log, dest))
		}

		close(testDone)
	}()

	// Should create channel and then close it since its stale
	assert.Equal(t, dest, <-client.createStreamChan, "should create stream")

	// 1st publish
	input := <-client.putLogsChan
	assert.Len(t, input.LogEvents, 1, "should get one log event")
	assert.Equal(t, "message0", *input.LogEvents[0].Message, "message should be correct")

	// Should create channel again since it was closed earlier
	assert.Equal(t, dest, <-client.createStreamChan, "should create stream again")

	// 2nd publish
	input = <-client.putLogsChan
	assert.Len(t, input.LogEvents, 1, "should get one log event")
	assert.Equal(t, "message1", *input.LogEvents[0].Message, "message should be correct")

	select {
	case <-ctx.Done():
		assert.Fail(t, "test timed out")
	case <-testDone:
	}
}

func TestMultipleStreams(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	client := mockCloudWatchLogsClient{
		createStreamChan: make(chan Destination),
		putLogsChan:      make(chan *cwlogs.PutLogEventsInput),
	}

	config := Config{
		MaxPublishingInterval: time.Millisecond * 10,
		BatchSize:             4,
		StreamTrackRetention:  time.Hour,
	}
	publisher, err := NewWithCWLClient(ctx, config, client)
	require.NoError(t, err, "creating publisher should not error")
	defer publisher.Close()

	now := time.Now()

	streamA := Destination{LogGroup: "group1", LogStream: "A"}
	streamB := Destination{LogGroup: "group1", LogStream: "B"}
	streamC := Destination{LogGroup: "group2", LogStream: "C"}

	// 4 messages from each of (group1, A) (group1, B) & (group2, C)
	inputs := []LogAndDestination{
		{Stream: streamA, Log: Log{"log message #1 A", now}},
		{Stream: streamC, Log: Log{"log message #1 C", now}},
		{Stream: streamB, Log: Log{"log message #1 B", now}},

		{Stream: streamC, Log: Log{"log message #2 C", now.Add(1 * time.Minute)}},
		{Stream: streamB, Log: Log{"log message #2 B", now.Add(1 * time.Minute)}},
		{Stream: streamA, Log: Log{"log message #2 A", now.Add(1 * time.Minute)}},

		{Stream: streamB, Log: Log{"log message #3 B", now.Add(2 * time.Minute)}},
		{Stream: streamA, Log: Log{"log message #3 A", now.Add(2 * time.Minute)}},
		{Stream: streamC, Log: Log{"log message #3 C", now.Add(2 * time.Minute)}},

		{Stream: streamA, Log: Log{"log message #4 A", now.Add(3 * time.Minute)}},
		{Stream: streamB, Log: Log{"log message #4 B", now.Add(3 * time.Minute)}},
		{Stream: streamC, Log: Log{"log message #4 C", now.Add(3 * time.Minute)}},
	}

	testDone := make(chan struct{})
	go func() {
		streams := []Destination{}
		for i := 0; i < 3; i++ {
			newStream := <-client.createStreamChan
			streams = append(streams, newStream)
		}

		assert.ElementsMatch(t, []Destination{
			streamA,
			streamB,
			streamC,
		}, streams, "should attempt to create correct streams")

		logInputs := []cwlogs.PutLogEventsInput{}
		for i := 0; i < 3; i++ {
			inp := <-client.putLogsChan
			logInputs = append(logInputs, *inp)
		}

		for _, putLogInput := range logInputs {
			dst := Destination{*putLogInput.LogGroupName, *putLogInput.LogStreamName}
			for i, logEvent := range putLogInput.LogEvents {
				messageExpectation := fmt.Sprintf(
					"log message #%d %v",
					i+1,
					dst.LogStream,
				)

				timeExpectation := now.Add(time.Duration(int(time.Minute) * i))

				assert.Equal(t, messageExpectation, *logEvent.Message, "message should be correct and in correct stream")
				assert.Equal(t, timeExpectation.UnixMilli(), *logEvent.Timestamp, "timestamp should be correct and in ascending order")
			}
		}

		close(testDone)
	}()

	for _, input := range inputs {
		assert.NoError(
			t,
			publisher.Push(ctx, input.Log, input.Stream),
			"should not error on pushing log",
		)
	}

	select {
	case <-ctx.Done():
		assert.Fail(t, "test timed out")
	case <-testDone:
	}
}
