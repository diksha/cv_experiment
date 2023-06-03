package ffmpeg_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/bazelbuild/rules_go/go/runfiles"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
)

// withMJPEGSample runs a test with a generated mjpeg sample file, cleaning up after the test returns
func withMJPEGSample(t *testing.T, testfn func(t *testing.T, filename string)) {
	// make a temp directory to put the mjpeg file in
	tmpdir, err := os.MkdirTemp("", "")
	require.NoError(t, err, "must make temp directory")

	// clean up our temp directory when we are done
	defer func() {
		_ = os.Remove(tmpdir)
	}()

	filename := filepath.Join(tmpdir, "testfile.mjpeg")
	var flags []string
	flags = append(flags, ffmpeg.TestInput{
		Duration: 10 * time.Second,
		Width:    320,
		Height:   240,
		Fps:      1,
	}.FFmpegFlags()...)
	flags = append(flags, filename)
	cmd, err := ffmpeg.New(context.TODO(), flags...)
	require.NoError(t, err, "must construct ffmpeg command")
	require.NoError(t, cmd.Run(), "must run ffmpeg command successfully")

	testfn(t, filename)
}

func TestProbe(t *testing.T) {
	res, err := ffmpeg.Probe(context.TODO(), ffmpeg.InputAutodetect(mustGetTestMediaPath()))
	require.NoError(t, err, "probe should not error")

	assert.Equal(t, len(res.Streams), 1, "result should have one stream")
}

func TestProbeTestSource(t *testing.T) {
	testsrc := ffmpeg.TestInput{
		Width:    320,
		Height:   240,
		Duration: 10 * time.Second,
		Fps:      5,
	}

	res, err := ffmpeg.Probe(context.TODO(), testsrc)
	require.NoError(t, err, "probe should not error")

	assert.Equal(t, len(res.Streams), 1, "result should have one stream")
	assert.Equal(t, ffmpeg.CodecNameRawVideo, res.Streams[0].CodecName, "codec type should be rawvideo")
	assert.Equal(t, ffmpeg.CodecTypeVideo, res.Streams[0].CodecType, "codec type should be video")
}

func TestProbeH264Profile(t *testing.T) {
	testsrc := ffmpeg.InputAutodetect(mustGetTestMediaPath())

	res, err := ffmpeg.Probe(context.TODO(), testsrc)
	require.NoError(t, err, "probe should not error")

	assert.Len(t, res.Streams, 1, "should have 1 stream")
	assert.Equal(t, ffmpeg.CodecProfileH264Main, res.Streams[0].Profile, "should be main profile")
}

func TestProbeMJPEG(t *testing.T) {
	withMJPEGSample(t, func(t *testing.T, mjpegfile string) {
		res, err := ffmpeg.Probe(context.TODO(), ffmpeg.InputAutodetect(mjpegfile))
		require.NoError(t, err, "must probe mjpeg file successfully")

		require.Len(t, res.Streams, 1, "result should have one stream")
		require.Equal(t, ffmpeg.CodecNameMJPEG, res.Streams[0].CodecName, "codec type should be mjpeg")
	})
}

func TestProbeFrames(t *testing.T) {
	// Arrange
	videoFilePath, err := runfiles.Rlocation("artifacts_05_19_2023_example_incident_mp4/00069972-3cb8-4639-b790-ca84c63fb5b2_video.mp4")
	require.NoError(t, err, "must find test media")

	// Act
	res, err := ffmpeg.ProbeFrames(context.TODO(), ffmpeg.InputAutodetect(videoFilePath))
	require.NoError(t, err, "probe should not error")

	// Assert
	assert.Len(t, res.Frames, 113)
	assert.Equal(t, "video", res.Frames[1].MediaType, "MediaType should be correct")
	assert.Equal(t, 0, res.Frames[1].StreamIndex, "StreamIndex should be correct")
	assert.Equal(t, 0, res.Frames[1].KeyFrame, "KeyFrame should be correct")
	assert.Equal(t, 3200, res.Frames[1].PTS, "PTS should be correct")
	assert.Equal(t, "0.200000", res.Frames[1].PTSTime, "PTSTime should be correct")
	assert.Equal(t, 3200, res.Frames[1].PacketDTS, "PacketDTS should be correct")
	assert.Equal(t, "0.200000", res.Frames[1].PacketDTSTime, "PacketDTSTime should be correct")
	assert.Equal(t, 3200, res.Frames[1].BestEffortTimestamp, "BestEffortTimestamp should be correct")
	assert.Equal(t, "0.200000", res.Frames[1].BestEffortTimestampTime, "BestEffortTimestampTime should be correct")
	assert.Equal(t, "462706", res.Frames[1].PacketPosition, "PacketPosition should be correct")
	assert.Equal(t, "6380", res.Frames[1].PacketSize, "PacketSize should be correct")
	assert.Equal(t, 960, res.Frames[1].Width, "Width should be correct")
	assert.Equal(t, 720, res.Frames[1].Height, "Height should be correct")
	assert.Equal(t, "yuv420p", res.Frames[1].PixelFormat, "PixelFormat should be correct")
	assert.Equal(t, "B", res.Frames[1].PictureType, "PictureType should be correct")
	assert.Equal(t, 2, res.Frames[1].CodedPictureNumber, "CodedPictureNumber should be correct")
	assert.Equal(t, 0, res.Frames[1].DisplayPictureNumber, "DisplayPictureNumber should be correct")
	assert.Equal(t, 0, res.Frames[1].InterlacedFrame, "InterlacedFrame should be correct")
	assert.Equal(t, 0, res.Frames[1].TopFieldFirst, "TopFieldFirst should be correct")
	assert.Equal(t, 0, res.Frames[1].RepeatPict, "RepeatPict should be correct")
	assert.Equal(t, "left", res.Frames[1].ChromaLocation, "ChromaLocation should be correct")
}
