package transcoder

import (
	"time"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
)

// RTSPInput returns flags appropriate for configuring an RTSP input
func RTSPInput(uri string) ffmpeg.Input {
	return ffmpeg.InputFlags([]string{
		"-rtsp_transport", "tcp",
		"-allowed_media_types", "video",
		"-f", "rtsp",
		"-i", uri,
	})
}

// TestInput returns flags appropriate for configuring a lavfi testsrc input
func TestInput(width, height, fps int, realtime bool, duration time.Duration) ffmpeg.Input {
	return ffmpeg.TestInput{
		Realtime: realtime,
		Width:    width,
		Height:   height,
		Fps:      fps,
		Duration: duration,
	}
}
