package transcoder

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog/log"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
	"github.com/voxel-ai/voxel/go/edge/metricsctx"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/kvspusher"
)

const (
	defaultTimeout         = 30 * time.Second
	defaultSegmentDuration = 10 * time.Second
)

// Mode is one of quicksync, cuda, or software
type Mode int

// This is the set of valid modes
const (
	ModeUndefined Mode = iota
	ModeQuicksync
	ModeCuda
	ModeSoftware
)

func (tm Mode) String() string {
	switch tm {
	case ModeCuda:
		return "cuda"
	case ModeQuicksync:
		return "quicksync"
	case ModeSoftware:
		return "software"
	case ModeUndefined:
		fallthrough
	default:
		return "undefined"
	}
}

// DefaultFilter is the default filter that should be applied to all streams
var DefaultFilter = NewSimpleFilter(
	// the following is quite hard to read, please read fpsfilter_test.go to get a better understanding of how this works.
	"select='eq(n,0)+if(gte(t-prev_selected_t,0.180)+gte(t-ld(0),0.370)+gte(t-ld(1),0.570)+gte(t-ld(2),0.780)+gte(t-ld(3),1.0),st(3,ld(2))+st(2,ld(1))+st(1,ld(0))+st(0,prev_selected_t)+1)'",
	// the following setpts filter sets frames to be assigned timestamps relative to the wall clock
	"setpts=(PTS-STARTPTS)+RTCSTART/(TB*1000000)",
)

// DefaultFlags is the default set of ffmpeg flags used by all modes
var DefaultFlags = FFmpegFlags{
	Global: Flags{
		// never ask to overwrite, this is useful when doing software/testing encodes and otherwise
		// isn't harmful really so we leave it in
		"-y",
		// this flag works with the -r/-fpsmax flag and ensures that we do not dup frames, only drop or pass through
		"-vsync", "vfr",
		// disable the standard stats output since we use a progress endpoint
		// this cleans up the ffmpeg output considerably and makes it easier to read
		"-nostats",
	},
	Filter: DefaultFilter,
	Encode: Flags{
		// drop audio frames
		"-an",
		// this instructs the encoder that the max framerate should be 5fps and should drop
		// any excess frames to achieve this value. unfortuantely the drop logic depends on the encoder time base being 1/fps
		// and since we're setting it to 1/1000, this doesn't actually drop or limit frames for us. Even so, the flag appears to be
		// required for the muxer not to get confused about the rate of frames produced by the encoder so we keep it in place.
		"-fpsmax", "5",
		// this ensures that our timestamp resolution is set to a value high enough to preserve the resolution of input timestamps
		// without this flag, timestamps will be set to 1/fps (so 1/5 in our case) which means timestamps have at best a 200ms resolution.
		// setting the time base to 1:1000 means we maintain a 1ms resolution
		"-enc_time_base", "1:1000",
	},
}

// Config holds the configuraiton parameters used to set up a Transcoder
type Config struct {
	Mode Mode

	Input         ffmpeg.Input
	EncoderConfig EncoderConfig

	Timeout  time.Duration
	LogLevel string
}

// Layout is a video layout, mostly just portrait or landscape
type Layout int

// The list of video layout configurations
const (
	LayoutAuto = iota
	LayoutPortrait
	LayoutLandscape
)

// EncoderConfig holds the values required to actually configure ffmpeg for quicksync/cuda/software encodes
type EncoderConfig struct {
	VideoBitrateKBPS int
	SegmentDuration  time.Duration

	Scaler ScalerConfig
	Remap  RemapConfig
}

// ScalerConfig holds configuration data for the various software/hardware scalers
type ScalerConfig struct {
	Enabled    bool
	Resolution int
	Layout     Layout
}

// RemapConfig holds configuration data for ffmpeg's remap filter
type RemapConfig struct {
	Enabled  bool
	PGMXPath string
	PGMYPath string
}

// Transcoder holds the handle to the ffmpeg command required to transcode
// and upload content to Kinesis Video. Transcoder should only be constructed with NewTranscoder
type Transcoder struct {
	Config Config

	progress atomic.Value
	receiver *receiver

	inputStream ffmpeg.ProbeStreamResult
	probeResult ffmpeg.ProbeResult
	ffCmd       *ffmpeg.Cmd
}

// Error is a transcoder error with additional log data
type Error struct {
	Logs string
	Err  error
}

func (e *Error) Error() string {
	return e.Err.Error()
}

func (e *Error) Unwrap() error {
	return e.Err
}

// Init initializes a Transcode by probing the input and validating the configuration
func Init(ctx context.Context, config Config) (*Transcoder, error) {
	if config.Timeout == 0 {
		config.Timeout = defaultTimeout
	}

	trans := &Transcoder{
		Config: config,
	}

	if err := trans.probeInput(ctx); err != nil {
		return nil, fmt.Errorf("transcoder input probe failed: %w", err)
	}

	trans.configureScaler()

	if err := trans.startReceiver(ctx); err != nil {
		return nil, fmt.Errorf("receiver start failure: %w", err)
	}

	if err := trans.prepareCmd(ctx); err != nil {
		return nil, fmt.Errorf("failed to initalize transcoder: %w", err)
	}

	return trans, nil
}

func (tr *Transcoder) prepareCmd(ctx context.Context) error {
	flags := DefaultFlags

	// append input flags
	flags.Input.Append(tr.Config.Input.FFmpegFlags()...)

	segmentDuration := tr.Config.EncoderConfig.SegmentDuration
	if segmentDuration == 0 {
		segmentDuration = defaultSegmentDuration
	}

	// append encoder flags
	flags.Encode.Append(
		"-b:v", fmt.Sprintf("%dk", tr.Config.EncoderConfig.VideoBitrateKBPS),
		// set the maximum bitrate to 4x the target to allow for spikes
		"-maxrate", fmt.Sprintf("%dk", tr.Config.EncoderConfig.VideoBitrateKBPS*4),
		// enforce keyframe interval
		"-force_key_frames", fmt.Sprintf("expr:if(isnan(prev_forced_t),1,gte(t,prev_forced_t+%.4f))", segmentDuration.Seconds()),
	)

	// append output flags
	flags.Output.Append(
		"-f", "segment",
		strings.TrimSuffix(tr.receiver.Endpoint(), "/")+"/chunk%d.mkv",
	)

	switch tr.Config.Mode {
	case ModeQuicksync:
		if err := SetQuicksyncFlags(&flags, tr.inputStream, tr.Config.EncoderConfig); err != nil {
			return fmt.Errorf("failed to set quicksync ffmpeg flags: %w", err)
		}
	case ModeCuda:
		if err := SetCudaFlags(&flags, tr.inputStream.CodecName, tr.Config.EncoderConfig); err != nil {
			return fmt.Errorf("failed to set cuda ffmpeg flags: %w", err)
		}
	case ModeSoftware:
		if err := SetSoftwareFlags(&flags, tr.Config.EncoderConfig); err != nil {
			return fmt.Errorf("failed to set software ffmpeg flags: %w", err)
		}
	default:
		return fmt.Errorf("invalid transcoder mode %v", tr.Config.Mode)
	}

	// construct the ffmpeg command
	ffCmd, err := ffmpeg.New(ctx, flags.CmdArgs()...)
	if err != nil {
		return fmt.Errorf("failed to create ffmpeg cmd: %w", err)
	}

	// set the loglevel and pdeathsig on supported platforms (ensuring ffmpeg is killed if this process dies)
	ffCmd.LogLevel = tr.Config.LogLevel
	setPdeathsig(ffCmd.Cmd)

	tr.ffCmd = ffCmd
	return nil
}

func (tr *Transcoder) stopReceiver() {
	// we set the shutdown timeout for the receiver to something longer than one segment
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	_ = tr.receiver.Shutdown(ctx)
}

// Start will start the ffmpeg processes
func (tr *Transcoder) Start(ctx context.Context) (err error) {
	// allow up to 60 seconds for startup
	timeout := time.NewTimer(60 * time.Second)
	defer timeout.Stop()

	log.Ctx(ctx).Info().Str("cmd", tr.ffCmd.String()).Msg("starting ffmpeg")

	// run Start() in a goroutine in case of a hang (which we have observed before)
	startErr := make(chan error, 1)
	go func() {
		startErr <- tr.ffCmd.Start()
	}()

	select {
	case err := <-startErr:
		// run Start() in a goroutine in case of failure
		if err != nil {
			// stop the receiver on startup failures and then return an error
			tr.stopReceiver()
			dumpLogs(ctx, tr.ffCmd.Output())
			return fmt.Errorf("ffmpeg start error: %w", err)
		}
	case <-timeout.C:
		dumpLogs(ctx, tr.ffCmd.Output())
		return fmt.Errorf("timeout occurred while waiting for ffmpeg to start")
	}

	go dumpLogs(ctx, tr.ffCmd.Output())

	// we have started the transcoder successfully, now set up to stop the receiver when ffmpeg exits
	go func() {
		<-tr.ffCmd.Done()
		tr.stopReceiver()
	}()

	defer func() {
		if err != nil && tr.ffCmd.Process != nil {
			// kill ffmpeg if we error out
			_ = tr.ffCmd.Process.Kill()
		}
	}()

	select {
	case progress := <-tr.ffCmd.Progress():
		tr.setProgress(progress)
	case <-timeout.C:
		return fmt.Errorf("timed out waiting for ffmpeg progress")
	case <-tr.ffCmd.Done():
		if tr.ffCmd.Err() == nil {
			return fmt.Errorf("ffmpeg ended before startup completed")
		}
		return fmt.Errorf("ffmpeg exited while waiting for initial status: %w", tr.ffCmd.Err())
	}

	return nil
}

// Progress returns the latest ffmpeg progress value, or a zero value if none have been received.
// This function is safe to call from any goroutine
func (tr *Transcoder) Progress() ffmpeg.Progress {
	if val, ok := tr.progress.Load().(ffmpeg.Progress); ok {
		return val
	}
	return ffmpeg.Progress{}
}

func (tr *Transcoder) setProgress(progress ffmpeg.Progress) {
	tr.progress.Store(progress)
}

// Fragments returns a channel from which video fragments can be received. The behavior of this
// function is undefined when called before Start()
func (tr *Transcoder) Fragments() <-chan *kvspusher.Fragment {
	return tr.receiver.Fragments()
}

// CmdString returns a string representation of this transcoder's ffmpeg cmd
func (tr *Transcoder) CmdString() string {
	return tr.ffCmd.String()
}

// ProbeResult returns the probe result retrieved during Init
func (tr *Transcoder) ProbeResult() ffmpeg.ProbeResult {
	return tr.probeResult
}

// Wait will wait for the currently running transcoder process to complete. The behavior of wait
// when called before Start() is undefined, as is the behavior of Wait() called more than once
func (tr *Transcoder) Wait(ctx context.Context) error {
	// check for timeouts on this interval
	timeoutTicker := time.NewTicker(1 * time.Second)

	lastProgressTime := time.Now()
	for {
		select {
		case progress := <-tr.ffCmd.Progress():
			// emit a zero frames transcoded message when we stall so this shows up in the metrics
			// also drops progress updates that go backwards which seem to sometimes happen at the end
			if progress.Frame <= tr.Progress().Frame {
				metricsctx.Publish(ctx, "FrameTranscoded", time.Now(), metricsctx.UnitCount, float64(0), nil)
				break
			}

			// compute the number of frames since the last progress and update stored progress
			frameDiff := progress.Frame - tr.Progress().Frame
			lastProgressTime = time.Now()
			tr.setProgress(progress)

			metricsctx.Publish(ctx, "FrameTranscoded", time.Now(), metricsctx.UnitCount, float64(frameDiff), nil)
		case <-tr.ffCmd.Done():
			// ffmpeg exited, dump logs and exit
			if tr.ffCmd.Err() == nil {
				return nil
			}

			return fmt.Errorf("ffmpeg exited: %w", tr.ffCmd.Err())
		case <-timeoutTicker.C:
			if time.Since(lastProgressTime) > tr.Config.Timeout {
				_ = tr.ffCmd.Process.Kill()

				return fmt.Errorf("ffmpeg exited: %w", tr.ffCmd.Err())
			}
		}
	}
}

// Kill kills the currently running transcoder process
func (tr *Transcoder) Kill() error {
	err := tr.ffCmd.Process.Kill()
	if err != nil {
		return fmt.Errorf("error killing transcode process: %w", err)
	}
	return nil
}

func (tr *Transcoder) startReceiver(ctx context.Context) error {
	rec, err := startReceiver(ctx)
	if err != nil {
		return err
	}
	tr.receiver = rec
	return nil
}

func (tr *Transcoder) probeInput(ctx context.Context) error {
	// making a sub-context here to force the probe process to time out
	// as it appears to sometimes hang
	ctx, cancel := context.WithTimeout(ctx, 90*time.Second)
	defer cancel()

	res, err := ffmpeg.Probe(ctx, tr.Config.Input)
	if err != nil {
		return fmt.Errorf("ffprobe error: %w", err)
	}

	tr.probeResult = *res

	// find the first video stream and use that
	for _, s := range res.Streams {
		if s.CodecType == ffmpeg.CodecTypeVideo {
			tr.inputStream = s
			// found a valid stream
			return nil
		}
	}

	return fmt.Errorf("failed to find video stream in input content")
}

func (tr *Transcoder) configureScaler() {
	stream := tr.inputStream
	scaler := &tr.Config.EncoderConfig.Scaler
	// do nothing if the scaler is disabled
	if !scaler.Enabled {
		return
	}

	// determine the scaler layout
	if scaler.Layout == LayoutAuto {
		if stream.Width >= stream.Height {
			scaler.Layout = LayoutLandscape
		} else {
			scaler.Layout = LayoutPortrait
		}
	}

	// disable the scaler if our resolution is already below target
	switch scaler.Layout {
	case LayoutLandscape:
		if stream.Height <= scaler.Resolution {
			scaler.Enabled = false
		}
	case LayoutPortrait:
		if stream.Width <= scaler.Resolution {
			scaler.Enabled = false
		}
	}
}

func dumpLogs(ctx context.Context, ch <-chan string) {
	for line := range ch {
		log.Ctx(ctx).Info().Str("ctx", "ffmpeg").Msg(line)
	}
}
