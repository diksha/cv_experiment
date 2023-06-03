// transcoder provides an ffmpeg wraper for Voxel's edge devices
package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/cloudwatch"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	"github.com/voxel-ai/voxel/go/core/aws/iotcredentials"
	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
	"github.com/voxel-ai/voxel/go/core/utils/iorate"
	"github.com/voxel-ai/voxel/go/core/utils/retry"
	"github.com/voxel-ai/voxel/go/edge/metricsctx"
	edgeconfigpb "github.com/voxel-ai/voxel/protos/edge/edgeconfig/v1"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/fish2persp"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/kvspusher"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/transcoder"
)

const (
	// We set a very long fragment upload timeout for slow clients
	// if we begin falling behind the queue will drop fragments naturally
	// if this value is set too short we will never achieve a fragment upload
	// when clients are slow, and we would prefer to receive some fragments with gaps
	// rather than no fragments from slow clients
	putMediaFragmentTimeout = 1 * time.Minute
)

// App runs a transcdoer which publishes to kinesis video with the passed in Config
type App struct {
	Config Config

	// Debug holds various overrides used for debugging and tests
	Debug AppDebug

	streamConfig *edgeconfigpb.StreamConfig
	awsConfig    aws.Config
	pusher       *kvspusher.Client
	tmpdir       string
	transcoder   *transcoder.Transcoder
}

// AppDebug holds various overrides used for debugging and tests
type AppDebug struct {
	// allows to override the built-in kvs pusher for testing purposes
	KinesisVideoAPI kvspusher.KinesisVideoAPI

	// overrides the default metrics interval
	MetricsInterval time.Duration

	// forces software transcoding mode, useful for testing/debugging
	ForceSoftware bool

	// overrides the ffmpeg/ffprobe input
	Input ffmpeg.Input

	// overrides the stream config rather than loading it from disk
	StreamConfig *edgeconfigpb.StreamConfig

	// overrides the aws config used for metrics and kinesis video
	AWSConfig *aws.Config
}

// Run executes this transcoder and returns nil only if the transcoder executed successfully
func (a *App) Run(ctx context.Context) error {
	// if we are just doing a smoketest, run the smoketest and exit
	if a.Config.Smoketest {
		return a.runSmoketest(ctx)
	}

	// this cleanup function currently only removes the temp directory created
	// by this app, but can be used for any cleanup that should happen when this app exits
	defer a.cleanup()

	ctx, err := a.configure(ctx)
	if err != nil {
		return fmt.Errorf("transcoder configuration error: %w", err)
	}

	if err := a.prepare(ctx); err != nil {
		return fmt.Errorf("transcoder initalization error: %w", err)
	}

	if err := a.run(ctx); err != nil {
		return fmt.Errorf("transcoder runtime error: %w", err)
	}

	return nil
}

func (a *App) runSmoketest(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	mode, err := getTranscoderMode(ctx, false)
	if err != nil {
		return fmt.Errorf("smoketest failed: %w", err)
	}

	log.Ctx(ctx).Info().Stringer("mode", mode).Msg("smoketest complete")

	return nil
}

func (a *App) getInput() ffmpeg.Input {
	if a.Debug.Input != nil {
		return a.Debug.Input
	} else if a.Config.DebugMode {
		// debug mode turns on a 720p 5fps realtime unbounded test source
		return transcoder.TestInput(1280, 720, 5, true, 0)
	}

	return transcoder.RTSPInput(a.streamConfig.RtspUri)
}

func (a *App) loadStreamConfig() error {
	if a.Debug.StreamConfig != nil {
		a.streamConfig = a.Debug.StreamConfig
		return nil
	}

	data, err := os.ReadFile(a.Config.StreamConfig)
	if err != nil {
		return fmt.Errorf("failed to read stream config file %q: %w", a.Config.StreamConfig, err)
	}

	var streamConfig edgeconfigpb.StreamConfig
	if err := protojson.Unmarshal(data, &streamConfig); err != nil {
		return fmt.Errorf("failed to unmarhsal %q: %w", a.Config.StreamConfig, err)
	}

	merged := proto.Clone(&DefaultStreamConfig).(*edgeconfigpb.StreamConfig)
	proto.Merge(merged, &streamConfig)

	a.streamConfig = merged
	return nil
}

func (a *App) loadAWSConfig(ctx context.Context) error {
	if a.Debug.AWSConfig != nil {
		a.awsConfig = *a.Debug.AWSConfig
		return nil
	}

	awsConfig, err := config.LoadDefaultConfig(ctx, config.WithRegion(a.Config.AWS.Region))
	if err != nil {
		return fmt.Errorf("failed to load default aws config: %w", err)
	}

	// if we have been given a credentials endpoint, try to use it
	if a.Config.IOT.CredsEndpoint != "" {
		iotCreds, err := iotcredentials.NewProvider(
			a.Config.IOT.CredsEndpoint,
			a.Config.IOT.ThingName,
			a.Config.IOT.RoleAlias,
			iotcredentials.WithCertificates(
				a.Config.IOT.Cert,
				a.Config.IOT.PrivKey,
				a.Config.IOT.RootCA,
			))
		if err != nil {
			return fmt.Errorf("failed to load iotcredentials.Provider: %w", err)
		}
		awsConfig.Credentials = aws.NewCredentialsCache(iotCreds)
	}

	a.awsConfig = awsConfig
	return nil
}

func (a *App) configure(baseCtx context.Context) (context.Context, error) {
	// this context is used for any startup related contexts that are not persisted
	ctx, cancel := context.WithTimeout(baseCtx, 15*time.Second)
	defer cancel()

	if err := a.loadAWSConfig(ctx); err != nil {
		return ctx, fmt.Errorf("failed to load aws configuration: %w", err)
	}

	if err := a.loadStreamConfig(); err != nil {
		return ctx, fmt.Errorf("failed to load stream configuration: %w", err)
	}

	if a.streamConfig.DebugLog {
		baseCtx = log.Ctx(baseCtx).Level(zerolog.DebugLevel).WithContext(baseCtx)
	} else {
		baseCtx = log.Ctx(baseCtx).Level(zerolog.InfoLevel).WithContext(baseCtx)
	}

	metricsConfig := metricsctx.Config{
		Client:    cloudwatch.NewFromConfig(a.awsConfig),
		Interval:  1 * time.Minute,
		Namespace: "voxel/edge/transcoder",
		DryRun:    a.Config.DebugMode,
		Dimensions: metricsctx.Dimensions{
			"EdgeUUID":   a.Config.IOT.ThingName,
			"StreamName": a.streamConfig.KinesisVideoStream,
		},
	}

	if a.Debug.MetricsInterval != 0 {
		metricsConfig.Interval = a.Debug.MetricsInterval
	}

	// apply a metrics publisher to the persistent context
	return metricsctx.WithPublisher(baseCtx, metricsConfig), nil
}

func (a *App) kill() {
	_ = a.transcoder.Kill()
}

func (a *App) prepare(ctx context.Context) error {
	metricsctx.Publish(ctx, "Init", time.Now(), metricsctx.UnitCount, float64(1), nil)

	logger := log.Ctx(ctx)
	logger.Info().Msg("checking transcoder mode")
	mode, err := getTranscoderMode(ctx, a.Debug.ForceSoftware || a.Config.DebugMode)
	if err != nil {
		return fmt.Errorf("failed to determine transcode mode: %w", err)
	}

	logger.Info().Stringer("mode", mode).Msg("mode check success, initializing kvs publisher")

	var kvspusherOpts []kvspusher.ClientOpt
	if a.Debug.KinesisVideoAPI != nil {
		kvspusherOpts = append(kvspusherOpts, kvspusher.WithKinesisVideoAPI(a.Debug.KinesisVideoAPI))
	}

	pusher, err := kvspusher.Init(ctx, a.streamConfig.KinesisVideoStream, a.awsConfig, kvspusherOpts...)
	if err != nil {
		return fmt.Errorf("failed to initialize kvspusher: %w", err)
	}
	a.pusher = pusher

	tmpdir, err := os.MkdirTemp("", "edge-transcoder")
	if err != nil {
		return fmt.Errorf("failed to construct edge-transcoder temp directory: %w", err)
	}

	a.tmpdir = tmpdir

	encoderConfig, err := prepareEncoderConfig(ctx, tmpdir, a.streamConfig)
	if err != nil {
		return fmt.Errorf("failed to prepare encoder config: %w", err)
	}

	logger.Debug().Interface("encoder_config", encoderConfig).Msg("loaded encoder config")

	logger.Info().Msg("initializing transcoder")

	logLevel := "error"
	if a.streamConfig.DebugLog {
		logger.Debug().Msg("setting ffmpeg log level to info")
		logLevel = "info"
	}
	if a.Config.FFmpeg.LogLevel != "" {
		logger.Info().Msg("overriding ffmpeg log level")
		logLevel = a.Config.FFmpeg.LogLevel
	}

	trans, err := transcoder.Init(ctx, transcoder.Config{
		Mode:          mode,
		Input:         a.getInput(),
		EncoderConfig: encoderConfig,
		LogLevel:      logLevel,
	})
	if err != nil {
		return fmt.Errorf("failed to initialize transcoder: %w", err)
	}

	a.transcoder = trans

	logger.Debug().Interface("transcoder_config", trans.Config).Msg("loaded transcoder config")
	logger.Debug().Interface("probe_result", trans.ProbeResult()).Msg("probe success")
	logger.Info().Msg("transcoder init complete")

	return nil
}

func (a *App) run(ctx context.Context) error {
	logger := log.Ctx(ctx)

	logger.Info().Msg("starting transcoder")
	logger.Debug().Str("cmd", a.transcoder.CmdString()).Msg("executing command")

	err := a.transcoder.Start(ctx)
	if err != nil {
		return fmt.Errorf("transcoder startup error: %w", err)
	}

	uploadDone := make(chan struct{})
	go func() {
		defer close(uploadDone)
		a.uploadFragments(ctx)
	}()

	printDone := make(chan struct{})
	printCtx, printCancel := context.WithCancel(ctx)
	defer printCancel()
	go func() {
		defer close(printDone)
		a.printStats(printCtx)
	}()

	metricsctx.Publish(ctx, "InitSuccess", time.Now(), metricsctx.UnitCount, float64(1), nil)
	logger.Info().Msg("entering transcoder run loop")
	err = a.transcoder.Wait(ctx)
	if err != nil {
		return fmt.Errorf("transcoder runtime error: %w", err)
	}

	printCancel()

	select {
	case <-uploadDone:
	case <-ctx.Done():
		return fmt.Errorf("error waiting for upload to complete: %w", ctx.Err())
	}

	select {
	case <-printDone:
	case <-ctx.Done():
		return fmt.Errorf("error waiting for print to complete: %w", ctx.Err())
	}

	return nil
}

func (a *App) cleanup() {
	if a.tmpdir != "" {
		_ = os.RemoveAll(a.tmpdir)
	}

	for range a.transcoder.Fragments() {
		// drain the fragments channel
	}
}

// utility functions

func (a *App) printStats(ctx context.Context) {
	// trunk-ignore(semgrep/trailofbits.go.nondeterministic-select.nondeterministic-select): doesn't matter here
	ticker := time.NewTicker(statsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			curProgress := a.transcoder.Progress()
			log.Ctx(ctx).Info().
				Int64("frame", curProgress.Frame).
				Float64("fps", curProgress.Fps).
				Time("out_time", curProgress.OutTimestamp).
				Msg("")
		case <-ctx.Done():
			return
		}
	}
}

var errFatalFragmentValidation = errors.New("invalid fragment")

func validateFragmentTimestamps(ctx context.Context, frag *kvspusher.Fragment, lastFragmentStartTime time.Time) error {
	allTimestamps := frag.GetAllTimestamps()
	startTimestampDuration, err := frag.MinTimestamp()
	if err != nil {
		return fmt.Errorf("failed to get fragment start timestamp: %w", err)
	}

	endTimestampDuration, err := frag.MaxTimestamp()
	if err != nil {
		return fmt.Errorf("failed to get fragment end timestamp: %w", err)
	}

	startTimestamp := time.Unix(0, startTimestampDuration.Nanoseconds())
	endTimestamp := time.Unix(0, endTimestampDuration.Nanoseconds())

	logger := log.Ctx(ctx).With().Durs("timestamps", allTimestamps).Time("start_time", startTimestamp).Time("end_time", endTimestamp).Logger()

	// Latency between the end time of the fragment and the processing time
	timestampLatency := time.Since(endTimestamp)
	metricsctx.Publish(ctx, "FragmentEndTimeLatency", time.Now(), metricsctx.UnitMilliseconds, float64(timestampLatency.Milliseconds()), nil)

	// Difference between the current and previous fragment start times
	fragmentStartTimeDelta := startTimestamp.Sub(lastFragmentStartTime)
	if !lastFragmentStartTime.IsZero() {
		metricsctx.Publish(ctx, "FragmentStartTimeDelta", time.Now(), metricsctx.UnitMilliseconds, float64(fragmentStartTimeDelta.Milliseconds()), nil)
	}

	// validate fragment is not in the future
	if timestampLatency < -5*time.Minute {
		logger.Error().Msg("fragment end time is in the future")
		return fmt.Errorf("fragment end time is %s in the future: %w", -timestampLatency, errFatalFragmentValidation)
	}

	// validate fragment starts after the previous fragment
	if fragmentStartTimeDelta <= 0 {
		logger.Error().Msg("fragment start time is before last fragment start time")
		return fmt.Errorf("fragment start time is not after last fragment start time: %w", errFatalFragmentValidation)
	}

	return nil
}

func (a *App) uploadFragment(ctx context.Context, fragment *kvspusher.Fragment, lastFragmentStartTime time.Time) error {
	logger := log.Ctx(ctx)

	// Remove unnecessary data from the fragment
	fragment = kvspusher.SimplifyFragment(fragment)

	// Metrics and measurements
	duration, err := fragment.Duration()
	if err != nil {
		return fmt.Errorf("failed to get fragment duration: %w", err)
	}

	fragmentDroppedCount := 0
	fragmentInvalidCount := 0
	fragmentUploadFailureCount := 0
	defer func() {
		metricsctx.Publish(ctx, "FragmentDuration", time.Now(), metricsctx.UnitMilliseconds, float64(duration.Milliseconds()), nil)
		metricsctx.Publish(ctx, "FragmentDropped", time.Now(), metricsctx.UnitCount, float64(fragmentDroppedCount), nil)
		metricsctx.Publish(ctx, "FragmentInvalid", time.Now(), metricsctx.UnitCount, float64(fragmentInvalidCount), nil)
		metricsctx.Publish(ctx, "FragmentUploadFailure", time.Now(), metricsctx.UnitCount, float64(fragmentUploadFailureCount), nil)
	}()

	// validate the fragment timestamps
	if err := validateFragmentTimestamps(ctx, fragment, lastFragmentStartTime); err != nil {
		fragmentInvalidCount = 1
		fragmentDroppedCount = 1
		return fmt.Errorf("invalid fragment timestamps: %w", err)
	}

	fragdata, err := serializeFragment(fragment)
	if err != nil {
		fragmentDroppedCount = 1
		return fmt.Errorf("failed to marshal mkv fragment for kvs: %w", err)
	}

	rate := iorate.Unlimited
	if len(a.transcoder.Fragments()) == 0 {
		rate = iorate.ByteRate(float64(len(fragdata)) / duration.Seconds() * 1.1)
	}

	// sanity check our rate a bit. the rate calculation above
	// can go haywire if the duration of a fragment is very short
	// either because it is only one frame or because of any metadata
	// error. our best choice here is just to unset the rate
	if rate < 0 || rate > 10*iorate.MBPS {
		rate = iorate.Unlimited
	}

	start := time.Now()
	err = a.putMediaWithRetries(ctx, fragdata, rate)
	// generally we want to avoid splitting err = and error checking lines but we really need
	// to capture the duration here so we have to assign it. we will report it after the error check
	dur := time.Since(start)
	metricsctx.Publish(ctx, "FragmentUploadDuration", time.Now(), metricsctx.UnitMilliseconds, float64(dur.Milliseconds()), nil)
	if err != nil {
		fragmentUploadFailureCount = 1
		fragmentDroppedCount = 1
		logger.Error().Err(err).Dur("duration", dur).Msg("kvs upload failure")

		return fmt.Errorf("failed to upload fragment: %w", err)
	}

	logger.Debug().Dur("duration", dur).Msg("kvs upload success")
	return nil
}

func (a *App) uploadFragments(ctx context.Context) {
	lastSuccessTime := time.Now()
	var lastFragmentStartTime time.Time

	logger := log.Ctx(ctx)
	for frag := range a.transcoder.Fragments() {
		err := a.uploadFragment(ctx, frag, lastFragmentStartTime)
		if err == nil {
			lastSuccessTime = time.Now()
		} else if errors.Is(err, errFatalFragmentValidation) {
			logger.Error().Err(err).Msg("failed to upload fragment (fatal)")
			a.kill()
			return
		} else {
			logger.Error().Err(err).Msg("failed to upload fragment")
		}

		curFragmentStart, err := frag.MinTimestamp()
		if err != nil {
			logger.Error().Err(err).Msg("failed to get fragment start time")
			continue
		}
		lastFragmentStartTime = time.Unix(0, curFragmentStart.Nanoseconds())

		if time.Since(lastSuccessTime) > 10*time.Minute {
			logger.Error().Msg("Last fragment upload was over 10 minutes ago, killing transcoder and restarting")
			a.kill()
			return
		}
	}
}

func (a *App) putMediaWithRetries(ctx context.Context, data []byte, rate iorate.ByteRate) error {
	// set the upload timeout on the outer context
	ctx, cancel := context.WithTimeout(ctx, putMediaFragmentTimeout)
	defer cancel()

	logger := log.Ctx(ctx)

	err := retry.Exponential{
		Initial:  10 * time.Millisecond,
		MaxDelay: 10 * time.Second,
		Log: func(err error) {
			logger.Warn().Err(err).Msg("kvs upload retry")
		},
	}.Do(ctx, func(ctx context.Context) error {
		// inner timeout and outer timeout are the same, we enable retries just
		// so that a random disconnect does not cause a fragment to be lost.
		ctx, cancel := context.WithTimeout(ctx, putMediaFragmentTimeout)
		defer cancel()

		defer func() {
			// always remove the rate limit for retries
			rate = iorate.Unlimited
		}()

		r, err := iorate.NewReader(io.NopCloser(bytes.NewReader(data)), rate)
		if err != nil {
			return fmt.Errorf("failed to create reader: %w", err)
		}

		logger.Debug().Stringer("ratelimit", rate).Msg("starting PutMedia request")
		if err = a.pusher.PutMedia(ctx, r, int64(len(data))); err != nil {
			return fmt.Errorf("failed to put media: %w", err)
		}
		return nil
	})

	if err != nil {
		return fmt.Errorf("putMedia retry failure: %w", err)
	}
	return nil
}

func serializeFragment(frag *kvspusher.Fragment) ([]byte, error) {
	var buf bytes.Buffer
	if err := kvspusher.WriteFragment(&buf, frag); err != nil {
		return nil, fmt.Errorf("failed to serialize mkv fragment: %w", err)
	}
	return buf.Bytes(), nil
}

func prepareEncoderConfig(ctx context.Context, tmpdir string, cfg *edgeconfigpb.StreamConfig) (transcoder.EncoderConfig, error) {
	encoderConfig := transcoder.EncoderConfig{
		VideoBitrateKBPS: int(cfg.VideoBitrateKbps),
		SegmentDuration:  time.Duration(float64(time.Second) * float64(cfg.SegmentDurationS)),
	}

	if cfg.GetScaler().GetEnabled() {
		encoderConfig.Scaler.Enabled = true
		resPixels, err := resolutionPixels(cfg.Scaler.Resolution)
		if err != nil {
			return encoderConfig, fmt.Errorf("failed to configure scaler: %w", err)
		}
		encoderConfig.Scaler.Resolution = resPixels
	}

	if cfg.GetDewarp().GetEnabled() {
		// first we generate the remap PGM and store the files for ffmpeg to be able to retrieve them
		params := cfg.GetDewarp().GetFish2PerspRemap()
		pgmdata, err := fish2persp.GenerateRemapPGM(ctx, params)
		if err != nil {
			return encoderConfig, fmt.Errorf("failed to generate remap pgm files: %w", err)
		}

		pgmxpath := filepath.Join(tmpdir, "remap_x.pgm")
		pgmypath := filepath.Join(tmpdir, "remap_y.pgm")

		if err := os.WriteFile(pgmxpath, pgmdata.X, 0644); err != nil {
			return encoderConfig, fmt.Errorf("failed to write %q: %w", pgmxpath, err)
		}

		if err := os.WriteFile(pgmypath, pgmdata.Y, 0644); err != nil {
			return encoderConfig, fmt.Errorf("failed to write %q: %w", pgmypath, err)
		}

		encoderConfig.Remap.Enabled = true
		encoderConfig.Remap.PGMXPath = pgmxpath
		encoderConfig.Remap.PGMYPath = pgmypath

		// check to see if the output of our remap is already scaled, if so we disable the scaler
		var remapResolution int
		if params.Persp.WidthPixels > params.Persp.HeightPixels {
			remapResolution = int(params.Persp.HeightPixels)
		} else {
			remapResolution = int(params.Persp.WidthPixels)
		}

		// if our output resolution is less than or equal to the scaler resolution, disable the scaler
		if encoderConfig.Scaler.Enabled && remapResolution <= encoderConfig.Scaler.Resolution {
			encoderConfig.Scaler.Enabled = false
		}
	}

	return encoderConfig, nil
}
