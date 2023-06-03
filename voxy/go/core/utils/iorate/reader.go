package iorate

import (
	"fmt"
	"io"
	"time"
)

// Reader is a rate limited reader which can wrap another reader
type Reader struct {
	r         io.ReadCloser
	rate      ByteRate
	tickCount int

	done   chan struct{}
	ticker *time.Ticker
}

// NewReader constructs a new rate limit reader, limited to the specified rate in bytes per second
func NewReader(r io.ReadCloser, rate ByteRate) (*Reader, error) {
	if rate < 0 {
		return nil, fmt.Errorf("invalid rate < 0 specified")
	}

	if rate > 100*GBPS {
		return nil, fmt.Errorf("invalid rate > 100 GBPS")
	}

	return &Reader{
		r:      r,
		rate:   rate,
		done:   make(chan struct{}),
		ticker: time.NewTicker(1 * time.Second),
	}, nil
}

// Read calls Read on the underlying at the maximum of the specified rate
func (r *Reader) Read(buf []byte) (int, error) {
	// Read after close returns an error
	if r.done == nil {
		return 0, io.ErrUnexpectedEOF
	}

	// Unlimited rate is a passthrough
	if r.rate == Unlimited {
		return r.r.Read(buf)
	}

	count := 0
	for len(buf) > 0 {
		// we have written all we can, wait for the next tick
		if r.tickCount >= int(r.rate) {
			select {
			case <-r.ticker.C:
			case <-r.done:
				return count, io.ErrUnexpectedEOF
			}
			r.tickCount = 0
		}

		// calculate the length of the next read
		readLen := int(r.rate) - r.tickCount
		if readLen > len(buf) {
			readLen = len(buf)
		}

		// perform the read, return any errors
		n, err := r.r.Read(buf[:readLen])
		r.tickCount += n
		count += n
		if err != nil {
			return count, err
		}

		// slide our read buffer forward past the bytes which have been read
		buf = buf[n:]
	}

	return count, nil
}

// Close closes this rate limiter as well as the underlying reader
func (r *Reader) Close() error {
	r.ticker.Stop()
	if r.done != nil {
		close(r.done)
		r.done = nil
	}
	return r.r.Close()
}
