package ffmpegbazel_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/go/core/utils/ffmpeg/ffmpegbazel"
)

func TestFind(t *testing.T) {
	err := ffmpegbazel.Find()
	require.NoError(t, err, "does not error")
}
