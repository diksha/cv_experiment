package iorate_test

import (
	"bytes"
	"io"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/go/core/utils/iorate"
)

func randomTestData(t *testing.T, count int64) []byte {
	src := rand.New(rand.NewSource(time.Now().UnixNano()))
	buf := &bytes.Buffer{}
	_, err := io.CopyN(buf, src, count)
	require.NoError(t, err, "should succesfully create test data")
	return buf.Bytes()
}

type nopWriteCloser struct {
	io.Writer
}

func (w *nopWriteCloser) Close() error { return nil }

func TestWriteRate(t *testing.T) {
	t.Parallel()
	// please see TestReadRate to understand the +1
	testdata := randomTestData(t, 301)
	ratelimitw := iorate.NewWriter(&nopWriteCloser{io.Discard}, 100*iorate.BPS)
	start := time.Now()

	n, err := ratelimitw.Write(testdata)
	require.NoError(t, err, "copy should not fail")
	require.EqualValues(t, len(testdata), n, "should write all bytes")

	elapsed := time.Since(start)
	// 301 bytes should be copied in around 3s. the first 100 bytes are copied immediately,
	// the next 100 at 1s, the next 100 at 2s, etc
	assert.InDelta(t, 3.0, elapsed.Seconds(), 0.1, "copying 301 bytes should take around 3s")
}

func TestWriteCorrect(t *testing.T) {
	t.Parallel()

	out := &bytes.Buffer{}
	testdata := randomTestData(t, 333)
	ratelimitw := iorate.NewWriter(&nopWriteCloser{out}, 100*iorate.BPS)

	n, err := ratelimitw.Write(testdata)
	require.NoError(t, err, "write should not fail")
	require.EqualValues(t, len(testdata), n, "should write all bytes")
	assert.Equal(t, testdata, out.Bytes(), "testdata and written data should match")
}

func TestHighWriteRateCopy(t *testing.T) {
	t.Parallel()
	// please see TestReadRate to understand the +1
	testdata := randomTestData(t, 500*1024*3+1)
	out := &bytes.Buffer{}
	writer := iorate.NewWriter(&nopWriteCloser{out}, 500*iorate.KBPS)

	pr, pw := io.Pipe()
	go func() {
		defer func() {
			_ = pw.Close()
		}()
		_, _ = io.Copy(pw, bytes.NewReader(testdata))
	}()

	start := time.Now()
	n, err := io.Copy(writer, pr)
	elapsed := time.Since(start)

	require.NoError(t, err, "write should not fail")
	assert.EqualValues(t, len(testdata), n, "correct number of bytes were written")
	assert.Equal(t, testdata, out.Bytes(), "written bytes should match")
	assert.InDelta(t, 3.0, elapsed.Seconds(), 0.1, "copying 500KB*3 at 500KBPS should take ~3s")
}

func TestWriterDoubleClose(t *testing.T) {
	limitW := iorate.NewWriter(&nopWriteCloser{io.Discard}, 1*iorate.BPS)
	require.NoError(t, limitW.Close(), "close does not error")
	require.NoError(t, limitW.Close(), "second close does not error or panic")
}

func TestWriterWriteAfterClose(t *testing.T) {
	testdata := randomTestData(t, 100)
	limitW := iorate.NewWriter(&nopWriteCloser{io.Discard}, 10*iorate.BPS)
	require.NoError(t, limitW.Close(), "close does not error")
	n, err := limitW.Write(testdata)
	assert.Error(t, err, "write should error")
	assert.EqualValues(t, 0, n, "write should write 0 bytes after close")
}

type writeRecorder struct {
	count int
}

func (wr *writeRecorder) Write(buf []byte) (int, error) {
	wr.count++
	return len(buf), nil
}

func TestWriterUnlimited(t *testing.T) {
	wr := &writeRecorder{}
	writer := iorate.NewWriter(&nopWriteCloser{wr}, iorate.Unlimited)
	n, err := writer.Write([]byte{1})
	assert.NoError(t, err, "write should not error")
	assert.EqualValues(t, 1, n, "write should write 1 byte")
	assert.EqualValues(t, 1, wr.count, "write should call write once")
}
