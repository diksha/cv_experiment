package logtailer_test

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/services/edge/metricsagent/lib/logtailer"
)

func withTempDir(t *testing.T, fn func(t *testing.T, dirname string)) {
	dirname, err := os.MkdirTemp("", "logtailer-watch-test")
	require.NoError(t, err, "must make temp dir for testing")
	defer func() {
		err := os.RemoveAll(dirname)
		require.NoError(t, err, "must delete temp dir after testing")
	}()

	fn(t, dirname)
}

func mustPrintf(t *testing.T, filename string, format string, args ...interface{}) {
	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	require.NoErrorf(t, err, "file %q must open", filename)
	defer func() {
		require.NoErrorf(t, f.Close(), "file %q must close", filename)
	}()

	_, err = fmt.Fprintf(f, format, args...)
	require.NoErrorf(t, err, "must write to file %q", filename)
}

func TestWatch(t *testing.T) {
	withTempDir(t, func(t *testing.T, tmpdir string) {
		watcher, err := logtailer.Watch(
			filepath.Join(tmpdir, "edge-*.log"),
		)
		require.NoError(t, err, "must create watcher")
		defer watcher.Stop()

		mustPrintf(t, filepath.Join(tmpdir, "edge-0.log"), "this is a logline\ntest\n")
		mustPrintf(t, filepath.Join(tmpdir, "nomatch-10.log"), "this line should not show up\n")

		watcher.Refresh()

		lines := make(map[string][]string)
		linesdone := make(chan struct{})
		go func() {
			defer close(linesdone)
			for line := range watcher.Lines() {
				lines[line.Filename] = append(lines[line.Filename], line.Text)

				// we are expecting two lines so tell the watcher to stop once we see that
				if len(lines[filepath.Join(tmpdir, "edge-0.log")]) >= 2 {
					watcher.Stop()
				}
			}
		}()

		// timeout after 1s, watcher.Stop() may be called twice but that's fine
		go func() {
			<-time.After(1 * time.Second)
			watcher.Stop()
		}()

		<-linesdone

		assert.Len(t, lines, 1, "should only have lines from one file")
		loglines := lines[filepath.Join(tmpdir, "edge-0.log")]
		assert.Len(t, loglines, 2, "the correct file should have 2 lines")
		assert.Equal(t, loglines[1], "test", "the lines should be correct")
	})
}
