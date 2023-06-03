package ffmpeg_test

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
)

const progressSample = `frame=127
fps=5.37
stream_0_0_q=-0.0
bitrate= 381.3kbits/s
total_size=1048576
out_time_us=22001000
out_time_ms=22001000
out_time=00:00:22.001000
dup_frames=0
drop_frames=0
speed=0.93x
progress=continue
frame=129
fps=5.33
stream_0_0_q=-0.0
bitrate= 374.5kbits/s
total_size=1048576
out_time_us=22401000
out_time_ms=22401000
out_time=00:00:22.401000
dup_frames=0
drop_frames=0
speed=0.926x
progress=continue
frame=132
fps=5.34
stream_0_0_q=-0.0
bitrate= 455.9kbits/s
total_size=1310720
out_time_us=23001000
out_time_ms=23001000
out_time=00:00:23.001000
dup_frames=1
drop_frames=2
speed=0.931x`

func TestReadProgress(t *testing.T) {
	// sample value only has 3 entries
	progressCh := make(chan ffmpeg.Progress, 4)
	buf := bytes.NewBuffer([]byte(progressSample))

	ffmpeg.ReadProgress(progressCh, buf)

	var progress []ffmpeg.Progress
	for p := range progressCh {
		progress = append(progress, p)
	}

	assert.EqualValues(t, 3, len(progress), "should get the correct number of progress updates")
	assert.EqualValues(t, 127, progress[0].Frame, "frame value of first progress entry should be correct")
	assert.EqualValuesf(t, 5.37, progress[0].Fps, "fps value of first progress entry should be correct")
	assert.Equal(t, "381.3kbits/s", progress[0].BitrateString, "bitrate value of first prgoress entry should be correct")
	assert.EqualValues(t, 1048576, progress[1].TotalSize, "total size value of second progress entry should be correct")
	assert.EqualValues(t, time.UnixMicro(23001000), progress[2].OutTimestamp, "out time value of third progress entry should be correct")
	assert.EqualValues(t, 1, progress[2].DupFrames, "dup frames value of third progress entry should be correct")
	assert.EqualValues(t, 2, progress[2].DropFrames, "drop frames value of third progress entry should be correct")
	assert.EqualValuesf(t, 0.931, progress[2].Speed, "speed value of third progress entry should be correct")
}

func TestStart(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	t.Log("starting transcode")
	transcode, err := ffmpeg.Start(ctx, []string{
		"-y",
		"-i", mustGetTestMediaPath(),
		"-c:v", "libx264",
		"-t", "30",
		"-f", "matroska",
		"/dev/null",
	}...)
	require.NoError(t, err, "StartTranscode should not fail")

	t.Log("waiting for output")
	err = transcode.Wait()
	if err != nil {
		var buf bytes.Buffer
		for line := range transcode.Output() {
			_, _ = buf.WriteString(line)
		}
		t.Log(buf.String())
	}
	require.NoError(t, err, "Wait should not error")

	t.Log("getting progress updates")
	var progress []ffmpeg.Progress
	for p := range transcode.Progress() {
		progress = append(progress, p)
	}

	require.Len(t, progress, 1, "should be able to retrieve exactly one progress update")
}
