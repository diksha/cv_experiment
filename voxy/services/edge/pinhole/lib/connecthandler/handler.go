// Package connecthandler provides a handler for HTTP 1.1 CONNECT requests
package connecthandler

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog/hlog"
	"github.com/rs/zerolog/log"
)

// New returns a new HTTP 1.1 CONNECT handler with an optional dialer
// If d is nil the default dialer (net.Dial) will be used
func New() http.Handler {
	return &handler{}
}

type handler struct {
	seq int32
}

// doFlush makes sure that all buffered bytes in clientRw are properly flushed in both directions
func (h *handler) doFlush(ctx context.Context, remote net.Conn, clientRw *bufio.ReadWriter) error {
	// we are going to pull any pre-buffered data out of the readwriter
	// and write it to the destination before proxying the connection
	//
	// the reason we do this is that we want to call io.Copy directly on the net.Conn
	// instances because they can take advantage of splice(2) for proxying that way
	// if we copy via the *bufio.ReadWriter this will not work and we will copy
	// all of the data to userspace, costing more cpu
	//
	//  https://go-review.googlesource.com/c/go/+/107715
	//
	writeFlushErr := make(chan error, 1)
	readFlushErr := make(chan error, 1)

	go func() {
		// flush any buffered data out to the client
		writeFlushErr <- clientRw.Flush()
	}()

	go func() {
		var err error
		defer func() {
			readFlushErr <- err
		}()

		var data []byte
		buffered := clientRw.Reader.Buffered()
		if buffered > 0 {
			data, err = clientRw.Reader.Peek(buffered)
			if err != nil {
				err = fmt.Errorf("failed to peek bytes from client: %w", err)
				log.Ctx(ctx).Debug().Err(err).Msg("failed to peek buffered bytes")
				return
			}

			_, err = remote.Write(data)
			if err != nil {
				err = fmt.Errorf("failed to write bytes to remote: %w", err)
				log.Ctx(ctx).Debug().Err(err).Msg("failed to write buffered bytes")
				return
			}
		}
	}()

	for i := 0; i < 2; i++ {
		select {
		case err := <-writeFlushErr:
			if err != nil {
				return fmt.Errorf("error flushing bytes to client: %w", err)
			}
		case err := <-readFlushErr:
			if err != nil {
				return fmt.Errorf("error flushing bytes to remote: %w", err)
			}
		}
	}

	return nil
}

func (h *handler) doCopy(ctx context.Context, dst, src net.Conn, errch chan<- error) {
	var err error
	var count int64

	// put this in its own defer so we are sure it always runs
	defer func() {
		errch <- err
	}()

	// always log completion
	defer func() {
		log.Ctx(ctx).Debug().
			Err(err).
			Int64("bytes", count).
			Stringer("src", src.RemoteAddr()).
			Stringer("dst", dst.RemoteAddr()).
			Msg("copy complete")
	}()

	// set some deadlines to avoid the risk of half-open connections
	// these cause read/write to always error on these by the specified time
	defer func() {
		_ = dst.SetDeadline(time.Now().Add(1 * time.Second))
		_ = src.SetDeadline(time.Now().Add(1 * time.Second))
	}()

	count, err = io.Copy(dst, src)
	if err != nil {
		err = fmt.Errorf("connection copy error: %w", err)
	}
}

func (h *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// set up a logger with a sequence id so we can figure out which messages belong to which request
	seq := atomic.AddInt32(&h.seq, 1)
	logger := hlog.FromRequest(r).With().Int32("seq", seq).Logger()

	// rewrap the request context so we have our context
	r = r.WithContext(logger.WithContext(r.Context()))

	logger.Debug().Msg("request")

	if r.Method != http.MethodConnect {
		logger.Error().Str("method", r.Method).Msg("invalid request method")
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	targetConn, err := net.Dial("tcp4", r.Host)
	if err != nil {
		logger.Error().Err(err).Msg("failed to dial target")
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer func() { _ = targetConn.Close() }()

	w.Header().Add("Transfer-Encoding", "identity")
	w.WriteHeader(http.StatusOK)
	hj, ok := w.(http.Hijacker)
	if !ok {
		logger.Error().Msg("http server does not support hijacking connetion")
		http.Error(w, "internal server error", http.StatusInternalServerError)
		return
	}

	clientConn, clientReadWriter, err := hj.Hijack()
	if err != nil {
		logger.Error().Err(err).Msg("http hijacking failed")
		return
	}
	defer func() { _ = clientConn.Close() }()

	logger.Debug().Msg("flushing")

	if err = h.doFlush(r.Context(), targetConn, clientReadWriter); err != nil {
		logger.Error().Err(err).Msg("failed to flush client connection")
		return
	}

	logger.Debug().Msg("established")

	errch := make(chan error, 2)
	go h.doCopy(r.Context(), clientConn, targetConn, errch)
	go h.doCopy(r.Context(), targetConn, clientConn, errch)

	logger.Debug().Msg("proxying")

	// consume the first error and log it as an error
	select {
	case err := <-errch:
		if err != nil {
			logger.Error().Err(err).Msg("unexpected error while proxying connection")
		}
	case <-r.Context().Done():
		logger.Error().Err(r.Context().Err()).Msg("force closing connection")
		return
	}

	select {
	case err := <-errch:
		logger.Debug().Err(err).Msg("second error while proxying connection")
	case <-r.Context().Done():
		logger.Error().Err(r.Context().Err()).Msg("force closing half-open connection")
		return
	}
}
