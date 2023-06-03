package ffmpeg

import (
	"fmt"
	"strings"
	"time"
)

// Input is an ffmpeg/ffprobe input
type Input interface {
	FFmpegFlags() []string
	FFprobeFlags() []string
}

// InputFlags is a set of arbitrary ffmpeg/ffprobe input flags
type InputFlags []string

// FFmpegFlags returns these flags to ffmpeg without making any modifications
func (f InputFlags) FFmpegFlags() []string {
	return f
}

// FFprobeFlags returns these flags to ffprobe without making any modifications
func (f InputFlags) FFprobeFlags() []string {
	return f
}

// InputAutodetect returns InputFlags configured to allow ffmpeg/ffprobe to autodetect the input from
// the url/filename/path/etc
func InputAutodetect(input string) Input {
	return InputFlags{"-i", input}
}

// TestInput produces flags that configure a lavfi/testsrc based test pattern input for ffmpeg/ffprobe
type TestInput struct {
	Realtime bool
	Duration time.Duration
	Width    int
	Height   int
	Fps      int
}

func (ti TestInput) flags() []string {
	params := []string{
		fmt.Sprintf("s=%dx%d", ti.Width, ti.Height),
		fmt.Sprintf("r=%d", ti.Fps),
	}

	if ti.Duration > 0 {
		params = append(params, fmt.Sprintf("d=%.3f", ti.Duration.Seconds()))
	}

	return []string{
		"-f", "lavfi",
		"-i", fmt.Sprintf("testsrc=%s,format=yuv420p", strings.Join(params, ":")),
	}
}

// FFmpegFlags returns test input flags compatible with ffmpeg
func (ti TestInput) FFmpegFlags() []string {
	if ti.Realtime {
		return append([]string{"-re"}, ti.flags()...)
	}
	return ti.flags()
}

// FFprobeFlags returns test input flags compatible with ffprobe
func (ti TestInput) FFprobeFlags() []string {
	return ti.flags()
}
