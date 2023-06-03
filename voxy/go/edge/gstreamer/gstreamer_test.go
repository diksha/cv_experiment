package gstreamer_test

import (
	"bytes"
	"io/ioutil"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/voxel-ai/voxel/go/edge/gstreamer"
)

const gstBufferLogSampleFile = "testdata/gst-buffer-log-sample.txt"

func mustLoadTestData() []byte {
	out, err := ioutil.ReadFile(gstBufferLogSampleFile)
	if err != nil {
		panic(err)
	}
	return out
}

func TestReadStats(t *testing.T) {
	statsCh := make(chan gstreamer.Stats, 51)
	gstreamer.ReadStats(statsCh, bytes.NewReader(mustLoadTestData()))

	// the input file has 50 lines but one line has an out of order timestamp, which this reader should skip
	assert.Equal(t, 49, len(statsCh), "stats channel should have the correct number of items")

	stats := <-statsCh
	t.Logf("first timestamp=%d", stats.LastBufferTimestamp)
	assert.Equal(t, time.Duration(2127610427), stats.LastBufferTimestamp, "the first stats item should have a correct timestamp")

	for stats = range statsCh {
	}

	t.Logf("last timestamp=%d", stats.LastBufferTimestamp)
	assert.Equal(t, time.Duration(10491968030), stats.LastBufferTimestamp, "last stats item should have a correct timestamp")

}
