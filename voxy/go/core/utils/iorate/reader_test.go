package iorate_test

import (
	"bytes"
	"io"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/go/core/utils/iorate"
)

func TestReadRate(t *testing.T) {
	t.Parallel()
	// we use 301 here due to the naive implementation. The first 100 bytes are written immediately
	// and then there is a 1s wait after. this means that for 300 bytes we would:
	//
	// write 100
	// sleep 1
	// write 100
	// sleep 1
	// write 100
	//
	// thus only waiting 2s for 300 bytes. using 301 forces one more sleep.
	testdata := randomTestData(t, 301)
	reader, err := iorate.NewReader(io.NopCloser(bytes.NewReader(testdata)), 100*iorate.BPS)
	require.NoError(t, err, "must create reader")

	start := time.Now()
	n, err := io.Copy(io.Discard, reader)
	elapsed := time.Since(start)
	require.NoError(t, err)
	assert.EqualValues(t, len(testdata), n, "copied the correct number of bytes")
	assert.InDelta(t, 3.0, elapsed.Seconds(), 0.1, "copying 301 bytes should take around 3s")
}

func TestReadCorrect(t *testing.T) {
	t.Parallel()
	testdata := randomTestData(t, 333)
	reader, err := iorate.NewReader(io.NopCloser(bytes.NewReader(testdata)), 100*iorate.BPS)
	require.NoError(t, err, "must create reader")

	out := &bytes.Buffer{}
	n, err := io.Copy(out, reader)
	require.NoError(t, err, "copy should not error")
	assert.EqualValues(t, len(testdata), n, "should copy the correct number of elements")
	assert.Equal(t, testdata, out.Bytes(), "copied data should be correct")
}

func TestReadHighRate(t *testing.T) {
	t.Parallel()
	// please see TestReadRate to understand the +1
	testdata := randomTestData(t, 500*1024*3+1)
	reader, err := iorate.NewReader(io.NopCloser(bytes.NewReader(testdata)), 500*iorate.KBPS)
	require.NoError(t, err, "must create reader")

	// we use an io pipe here to force io.Copy to use the simple implmentation which
	// only allocates a 32k buffer for the copy
	piper, pipew := io.Pipe()
	done := make(chan struct{})
	out := &bytes.Buffer{}
	go func() {
		defer func() {
			close(done)
			_ = piper.Close()
		}()
		_, _ = io.Copy(out, piper)
	}()

	start := time.Now()

	n, err := io.Copy(pipew, reader)
	_ = pipew.Close()

	elapsed := time.Since(start)
	<-done

	require.NoError(t, err, "copy should not fail")
	assert.EqualValues(t, len(testdata), n, "correct number of bytes were copied")
	assert.Equal(t, testdata, out.Bytes(), "output data matches input data")
	assert.InDelta(t, 3.0, elapsed.Seconds(), 0.1, "copying 500KB*3 at 500KBPS should take ~3s")
}

func TestReaderDoubleClose(t *testing.T) {
	limitR, err := iorate.NewReader(io.NopCloser(bytes.NewReader([]byte{})), 1*iorate.BPS)
	require.NoError(t, err, "must create reader")
	require.NoError(t, limitR.Close(), "close does not error")
	require.NoError(t, limitR.Close(), "a second close does not error or panic")
}

func TestReaderReadAfterClose(t *testing.T) {
	testdata := randomTestData(t, 1)
	reader, err := iorate.NewReader(io.NopCloser(bytes.NewReader(testdata)), 10*iorate.BPS)
	require.NoError(t, err, "must create reader")
	require.NoError(t, reader.Close(), "close does not error")
	n, err := io.Copy(io.Discard, reader)
	assert.Error(t, err, "read after close produces an error")
	assert.EqualValues(t, 0, n, "reads 0 bytes after close")
}

type readRecorder struct {
	readCount int
}

func (rr *readRecorder) Read(buf []byte) (int, error) {
	rr.readCount++
	return len(buf), nil
}

func TestReaderUnlimited(t *testing.T) {
	rr := &readRecorder{}
	reader, err := iorate.NewReader(io.NopCloser(rr), iorate.Unlimited)
	require.NoError(t, err, "must create reader")
	n, err := reader.Read([]byte{0})
	assert.NoError(t, err, "read should not error")
	assert.EqualValues(t, 1, rr.readCount, "read should call read once")
	assert.EqualValues(t, 1, n, "should read 1 byte")
}

func TestReaderInvalidRate(t *testing.T) {
	_, err := iorate.NewReader(nil, -1)
	assert.Error(t, err, "should fail on a rate < 0")
	_, err = iorate.NewReader(nil, 100*iorate.GBPS+iorate.ByteRate(1))
	assert.Error(t, err, "should fail on rate > 100 GBPS")
}
