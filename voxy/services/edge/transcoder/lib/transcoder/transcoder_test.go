package transcoder_test

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg/ffmpegbazel"
	"github.com/voxel-ai/voxel/go/edge/metricsctx"
	edgeconfigpb "github.com/voxel-ai/voxel/protos/edge/edgeconfig/v1"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/fish2persp"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/kvspusher"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/transcoder"
)

func init() {
	// attempt to find ffmpeg from bazel
	_ = ffmpegbazel.Find()
}

func testContext(timeout time.Duration) (context.Context, func()) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	ctx = metricsctx.WithPublisher(ctx, metricsctx.Config{
		DryRun: true,
	})
	return ctx, cancel
}

func writeTempMKVFragment(t *testing.T, fragment *kvspusher.Fragment) string {
	outf, err := os.CreateTemp("", "transcoder-sample-*.mkv")
	require.NoError(t, err, "must create temp mkv file")
	defer func() {
		// if we recover a panic, this failed so we just clean up
		if err := recover(); err != nil {
			_ = outf.Close()
			_ = os.Remove(outf.Name())
			panic(err)
		}
	}()

	require.NoError(t, kvspusher.WriteFragment(outf, fragment), "must write temp mkv file data")
	require.NoError(t, outf.Close(), "must close temp mkv file")

	return outf.Name()
}

func probeFragment(t *testing.T, fragment *kvspusher.Fragment) *ffmpeg.ProbeResult {
	filename := writeTempMKVFragment(t, fragment)
	defer func() {
		_ = os.Remove(filename)
	}()

	res, err := ffmpeg.Probe(context.Background(), ffmpeg.InputAutodetect(filename))
	require.NoError(t, err, "must probe segment data successfully")

	return res
}

func runTranscoder(t *testing.T, cfg transcoder.Config) *transcoder.Transcoder {
	ctx, cancel := testContext(10 * time.Second)
	defer cancel()

	tr, err := transcoder.Init(ctx, cfg)
	require.NoError(t, err, "transcoder should initialize succesfully")
	require.NoError(t, tr.Start(ctx), "transcoder should start successfully")
	require.NoError(t, tr.Wait(ctx), "transcoder should complete succesfully")
	return tr
}

func getFragments(tr *transcoder.Transcoder) []*kvspusher.Fragment {
	var fragments []*kvspusher.Fragment
	for frag := range tr.Fragments() {
		fragments = append(fragments, frag)
	}
	return fragments
}

// A basic functionality test of the software transcoder which ensures that
// video semgents are produced
func TestSoftwareTranscoder(t *testing.T) {
	cfg := transcoder.Config{
		Mode:  transcoder.ModeSoftware,
		Input: transcoder.TestInput(480, 360, 25, false, 30*time.Second),
		EncoderConfig: transcoder.EncoderConfig{
			VideoBitrateKBPS: 500,
		},
		LogLevel: "debug",
	}

	tr := runTranscoder(t, cfg)

	fragment := getFragments(tr)[0]
	res := probeFragment(t, fragment)

	require.Len(t, res.Streams, 1, "transcoded output should have one stream")
	assert.InDelta(t, 5, fpsFromFraction(t, res.Streams[0].AvgFrameRate), 0.5, "output framerate should be correct")
}

// A basic functionality test of the software transcoder which ensures that
// video semgents are produced
func TestShortFragments(t *testing.T) {
	cfg := transcoder.Config{
		Mode:  transcoder.ModeSoftware,
		Input: transcoder.TestInput(480, 360, 25, false, 30*time.Second),
		EncoderConfig: transcoder.EncoderConfig{
			VideoBitrateKBPS: 500,
			SegmentDuration:  1 * time.Second,
		},
		LogLevel: "debug",
	}

	tr := runTranscoder(t, cfg)

	assert.Len(t, tr.Fragments(), 30, "transcoded output should have 30 fragments")

	fragment := getFragments(tr)[0]
	res := probeFragment(t, fragment)

	require.Len(t, res.Streams, 1, "transcoded output should have one stream")
}

// calculates a floating point fps from an ffmpeg frame rate fractional string
// supports entries like "5/1","100000/19989"
func fpsFromFraction(t *testing.T, framerate string) float64 {
	vals := strings.Split(framerate, "/")
	require.Lenf(t, vals, 2, "framerate string %q must split into two values", framerate)

	num, err := strconv.ParseFloat(vals[0], 64)
	require.NoErrorf(t, err, "must parse numerator value %q", vals[0])

	den, err := strconv.ParseFloat(vals[1], 64)
	require.NoErrorf(t, err, "must parse denominator value %q", vals[1])

	return num / den
}

// ensure that the scaler properly downscales landscape video
func TestSoftwareTranscoderDownscaleLandscape(t *testing.T) {
	cfg := transcoder.Config{
		Mode:  transcoder.ModeSoftware,
		Input: transcoder.TestInput(1600, 900, 5, false, 5*time.Second),
		EncoderConfig: transcoder.EncoderConfig{
			Scaler: transcoder.ScalerConfig{
				Enabled:    true,
				Resolution: 720,
			},
		},
		LogLevel: "debug",
	}

	tr := runTranscoder(t, cfg)
	require.Len(t, tr.Fragments(), 1, "should have one fragment")

	fragment := getFragments(tr)[0]
	probeRes := probeFragment(t, fragment)

	require.Len(t, probeRes.Streams, 1, "output fragment should have one stream")
	stream := probeRes.Streams[0]
	assert.Equal(t, 1280, stream.Width, "output stream width should be correct")
	assert.Equal(t, 720, stream.Height, "output stream height should be correct")

	require.Len(t, probeRes.Streams, 1, "output must have one stream")
	assert.InDelta(t, 5, fpsFromFraction(t, probeRes.Streams[0].AvgFrameRate), 0.5, "fps should be between 4.5 and 5.5")
}

// ensure that the scaler properly downscales portrait video
func TestSoftwareTranscoderDownscalePortrait(t *testing.T) {
	cfg := transcoder.Config{
		Mode:  transcoder.ModeSoftware,
		Input: transcoder.TestInput(900, 1600, 5, false, 5*time.Second),
		EncoderConfig: transcoder.EncoderConfig{
			Scaler: transcoder.ScalerConfig{
				Enabled:    true,
				Resolution: 720,
			},
		},
		LogLevel: "debug",
	}

	tr := runTranscoder(t, cfg)
	require.Len(t, tr.Fragments(), 1, "should have one fragment")

	fragment := getFragments(tr)[0]
	probeRes := probeFragment(t, fragment)

	require.Len(t, probeRes.Streams, 1, "output fragment should have one stream")
	stream := probeRes.Streams[0]
	assert.Equal(t, 720, stream.Width, "output stream width should be correct")
	assert.Equal(t, 1280, stream.Height, "output stream height should be correct")
	assert.InDelta(t, 5, fpsFromFraction(t, stream.AvgFrameRate), 0.5, "fps should be between 4.5 and 5.5")
}

// ensure that the scaler *does not* upscale video
func TestSoftwareTranscoderNoUpscale(t *testing.T) {
	cfg := transcoder.Config{
		Mode:  transcoder.ModeSoftware,
		Input: transcoder.TestInput(480, 360, 5, false, 5*time.Second),
		EncoderConfig: transcoder.EncoderConfig{
			Scaler: transcoder.ScalerConfig{
				Enabled:    true,
				Resolution: 720,
			},
		},
		LogLevel: "debug",
	}

	tr := runTranscoder(t, cfg)
	require.Len(t, tr.Fragments(), 1, "should have one fragment")

	fragment := getFragments(tr)[0]
	probeRes := probeFragment(t, fragment)
	require.Len(t, probeRes.Streams, 1, "output fragment must have one stream")
	stream := probeRes.Streams[0]
	assert.Equal(t, 480, stream.Width, "output stream width must be correct")
	assert.Equal(t, 360, stream.Height, "output stream height must be correct")
}

func TestSoftwareTranscoderCmdString(t *testing.T) {
	cfg := transcoder.Config{
		Mode:          transcoder.ModeSoftware,
		Input:         transcoder.TestInput(480, 360, 5, false, 5*time.Second),
		EncoderConfig: transcoder.EncoderConfig{},
		LogLevel:      "debug",
	}

	tr := runTranscoder(t, cfg)
	assert.Len(t, tr.Fragments(), 1, "should have one fragment")
	assert.Greater(t, len(tr.CmdString()), 32, "cmd string should be populated")
}

func TestSoftwareTranscoderProbeResults(t *testing.T) {
	cfg := transcoder.Config{
		Mode:          transcoder.ModeSoftware,
		Input:         transcoder.TestInput(480, 360, 5, false, 5*time.Second),
		EncoderConfig: transcoder.EncoderConfig{},
		LogLevel:      "debug",
	}

	tr := runTranscoder(t, cfg)
	assert.Len(t, tr.ProbeResult().Streams, 1, "probe result should have one stream")
}

func TestSoftwareTranscoderProgress(t *testing.T) {
	cfg := transcoder.Config{
		Mode:          transcoder.ModeSoftware,
		Input:         transcoder.TestInput(480, 360, 5, false, 5*time.Second),
		EncoderConfig: transcoder.EncoderConfig{},
		LogLevel:      "debug",
	}

	tr := runTranscoder(t, cfg)
	assert.Greater(t, tr.Progress().Frame, int64(0), "should have transcoded at least one frame")
}

func writeTempPGM(data []byte) (string, error) {
	outf, err := os.CreateTemp("", "remap-*.pgm")
	if err != nil {
		return "", fmt.Errorf("failed to write temp pgm: %w", err)
	}
	defer func() {
		_ = outf.Close()
	}()

	if _, err := outf.Write(data); err != nil {
		return "", fmt.Errorf("failed to write temp pgm: %w", err)
	}

	return outf.Name(), outf.Close()
}

// test software scaling remap
func TestSoftwareTranscoderRemap(t *testing.T) {
	ctx := context.Background()
	pgmdata, err := fish2persp.GenerateRemapPGM(ctx, &edgeconfigpb.Fish2PerspRemap{
		Fish: &edgeconfigpb.Fish2PerspRemap_Fish{
			WidthPixels:   480,
			HeightPixels:  360,
			CenterXPixels: 480 / 2,
			CenterYPixels: 360 / 2,
			RadiusXPixels: 360 / 2,
			FovDegrees:    180,
		},
		Persp: &edgeconfigpb.Fish2PerspRemap_Persp{
			WidthPixels:  480,
			HeightPixels: 360,
		},
	})
	require.NoError(t, err)

	pgmxpath, err := writeTempPGM(pgmdata.X)
	require.NoError(t, err)
	defer func() {
		_ = os.Remove(pgmxpath)
	}()

	pgmypath, err := writeTempPGM(pgmdata.Y)
	require.NoError(t, err)
	defer func() {
		_ = os.Remove(pgmypath)
	}()

	cfg := transcoder.Config{
		Mode:  transcoder.ModeSoftware,
		Input: transcoder.TestInput(480, 360, 25, false, 30*time.Second),
		EncoderConfig: transcoder.EncoderConfig{
			VideoBitrateKBPS: 500,
			Remap: transcoder.RemapConfig{
				Enabled:  true,
				PGMXPath: pgmxpath,
				PGMYPath: pgmypath,
			},
		},
		LogLevel: "debug",
	}

	tr := runTranscoder(t, cfg)

	fragment := getFragments(tr)[0]
	res := probeFragment(t, fragment)

	require.Len(t, res.Streams, 1, "transcoded output should have one stream")
}

func TestSoftwareTranscoderError(t *testing.T) {
	// we are going to test with a bad remap config to trigger an error after ffmpeg starts
	cfg := transcoder.Config{
		Mode:  transcoder.ModeSoftware,
		Input: transcoder.TestInput(480, 360, 25, false, 30*time.Second),
		EncoderConfig: transcoder.EncoderConfig{
			VideoBitrateKBPS: 500,
			Remap: transcoder.RemapConfig{
				Enabled:  true,
				PGMXPath: "/dev/null",
				PGMYPath: "/dev/null",
			},
		},
		LogLevel: "debug",
	}

	ctx, cancel := testContext(10 * time.Second)
	defer cancel()

	tr, err := transcoder.Init(ctx, cfg)
	require.NoError(t, err, "transcoder should initialize succesfully")
	require.NoError(t, tr.Start(ctx), "transcoder should start successfully")
	err = tr.Wait(ctx)
	require.Error(t, err, "transcoder should complete with an error")
}
