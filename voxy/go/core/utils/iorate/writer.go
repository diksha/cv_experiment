package iorate

import (
	"io"
	"time"
)

// Writer is a rate limited writer, ensuring that only the specified number of bytes are written each second
type Writer struct {
	w         io.WriteCloser
	rate      ByteRate
	tickCount int

	done   chan struct{}
	ticker *time.Ticker
}

// NewWriter constructs a new rate limit writer which limits the write rate to the specified rate, with a resolution of 1s
func NewWriter(w io.WriteCloser, rate ByteRate) *Writer {
	return &Writer{
		w:      w,
		rate:   rate,
		done:   make(chan struct{}),
		ticker: time.NewTicker(1 * time.Second),
	}
}

// Write calls Write on the underlying writer at a maximum of the specified rate
func (w *Writer) Write(buf []byte) (int, error) {
	if w.done == nil {
		return 0, io.ErrUnexpectedEOF
	}

	// unlimited rate is a passthrough
	if w.rate == Unlimited {
		return w.w.Write(buf)
	}

	count := 0
	for len(buf) > 0 {
		// we have written all we can this tick, so just wait
		if w.tickCount >= int(w.rate) {
			select {
			case <-w.ticker.C:
			case <-w.done:
				return count, io.ErrUnexpectedEOF
			}
			w.tickCount = 0
		}

		// calculate the length of the next write
		writeLen := int(w.rate) - w.tickCount
		if writeLen > len(buf) {
			writeLen = len(buf)
		}

		// perform the write, return any errors
		n, err := w.w.Write(buf[:writeLen])
		w.tickCount += n
		count += n
		if err != nil {
			return count, err
		}

		// trim any written bytes off the front of the buffer
		buf = buf[n:]
	}
	return count, nil
}

// Close closes this rate limiter as well as the underlying writer
func (w *Writer) Close() error {
	w.ticker.Stop()
	if w.done != nil {
		close(w.done)
		w.done = nil
	}
	return w.w.Close()
}
