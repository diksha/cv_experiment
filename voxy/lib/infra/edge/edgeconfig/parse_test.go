package edgeconfig_test

import (
	"io/ioutil"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/lib/infra/edge/edgeconfig"
)

func mustReadTestdata(t *testing.T, filename string) string {
	out, err := ioutil.ReadFile(filename)
	require.NoErrorf(t, err, "must read test file %q", filename)
	return string(out)
}

func TestValid(t *testing.T) {
	validFiles, err := filepath.Glob("testdata/valid/*.yaml")
	require.NoError(t, err, "filepath loads valid files")
	require.True(t, len(validFiles) > 1, "should have more than 1 valid file")

	for _, filename := range validFiles {
		_, err := edgeconfig.ParseYAML(mustReadTestdata(t, filename))
		require.NoErrorf(t, err, "loading %q should not error", filename)
	}
}

func TestInvalid(t *testing.T) {
	invalidFiles, err := filepath.Glob("testdata/invalid/*.yaml")
	require.NoError(t, err, "filepath loads invalid files")
	require.True(t, len(invalidFiles) > 1, "should have more than 1 invalid file")

	for _, filename := range invalidFiles {
		_, err := edgeconfig.ParseYAML(mustReadTestdata(t, filename))
		require.Error(t, err, "loading %q should error", filename)
	}
}

func TestParse(t *testing.T) {
	cfg, err := edgeconfig.ParseYAML(mustReadTestdata(t, filepath.Join("testdata", "edgeconfig.yaml")))
	require.NoError(t, err, "good file must parse")

	assert.EqualValues(t, cfg.Version, 1, "good file must have correct version")
	require.Len(t, cfg.Streams, 1, "good file should have the correct number of streams")
	assert.Equal(t, "test-uri", cfg.Streams[0].RtspUri, "rtsp uri should be correct")
	assert.Equal(t, "test-stream", cfg.Streams[0].KinesisVideoStream, "kinesis stream should be correct")
}
