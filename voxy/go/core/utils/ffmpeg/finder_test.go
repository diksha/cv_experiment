package ffmpeg_test

import (
	"log"
	"testing"

	"github.com/bazelbuild/rules_go/go/tools/bazel"
	"github.com/stretchr/testify/assert"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg"
)

const (
	testMediaArtifactName = "office_cam_720p.mkv"
)

func init() {
	ffmpegBin, err := bazel.Runfile("ffmpeg")
	if err != nil {
		log.Fatalf("Failed to find ffmpeg bin: %v", err)
	}

	ffprobeBin, err := bazel.Runfile("ffprobe")
	if err != nil {
		log.Fatalf("Failed to find ffprobe bin: %v", err)
	}

	if err = ffmpeg.SetFFmpegPath(ffmpegBin); err != nil {
		log.Fatalf("Failed to set ffmpeg path: %v", err)
	}

	if err = ffmpeg.SetFFprobePath(ffprobeBin); err != nil {
		log.Fatalf("Failed to set ffprobe path: %v", err)
	}
}

func mustGetTestMediaPath() string {
	filepath, err := bazel.Runfile(testMediaArtifactName)
	if err != nil {
		log.Fatalf("Failed to find bazel runfiles: %v", err)
	}

	return filepath
}

func TestFindArtifacts(t *testing.T) {
	_, err := ffmpeg.FindFFprobe()
	assert.NoError(t, err, "this test requires ffprobe")

	_, err = ffmpeg.FindFFmpeg()
	assert.NoError(t, err, "this test requires ffmpeg")

	_ = mustGetTestMediaPath()
}
