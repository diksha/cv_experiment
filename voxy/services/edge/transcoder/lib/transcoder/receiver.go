package transcoder

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"goji.io"
	"goji.io/pat"

	"github.com/rs/zerolog/hlog"
	"github.com/rs/zerolog/log"

	"github.com/voxel-ai/voxel/go/edge/metricsctx"
	"github.com/voxel-ai/voxel/services/edge/transcoder/lib/kvspusher"
)

// Receiver accepts fragments from ffmpeg via http
type receiver struct {
	endpoint  string
	fragments chan *kvspusher.Fragment
	srv       *http.Server
	srvErr    error
	ctx       context.Context
}

func startReceiver(ctx context.Context) (*receiver, error) {
	// we default to buffering up to 60 fragments, which should be 10 minutes/~37.5 MB
	//     (500 Kbps) * (10 s) *  60 == 37.5 MB
	rec := &receiver{
		fragments: make(chan *kvspusher.Fragment, 60),
		ctx:       ctx,
	}

	// report the fragment queue length
	// we deped on context cancellation to clean this up
	go func() {
		// we're ignoring this semgrep because we don't really care if the ticker runs an extra time
		// trunk-ignore(semgrep/trailofbits.go.nondeterministic-select.nondeterministic-select)
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				metricsctx.Publish(ctx, "FragmentQueueLength", time.Now(), metricsctx.UnitCount, float64(len(rec.fragments)), nil)
				log.Ctx(ctx).Debug().Int("fragment_queue", len(rec.fragments)).Msg("")
			case <-ctx.Done():
				return
			}
		}
	}()

	if err := rec.startServer(ctx); err != nil {
		return nil, fmt.Errorf("failed to start http server: %w", err)
	}

	return rec, nil
}

// Endpoint returns the http url this receiver listens on
func (rec *receiver) Endpoint() string {
	return rec.endpoint
}

// Fragments returns a channel to receive inbound fragments on
func (rec *receiver) Fragments() <-chan *kvspusher.Fragment {
	return rec.fragments
}

func (rec *receiver) handleFragment(w http.ResponseWriter, r *http.Request) {
	logger := hlog.FromRequest(r)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusInternalServerError)
		return
	}

	frag, err := kvspusher.ReadFragment(bytes.NewReader(body))
	if err != nil {
		http.Error(w, fmt.Sprintf("invalid mkv data: %v", err), http.StatusBadRequest)
		return
	}

	select {
	case rec.fragments <- frag:
	default:
		metricsctx.Publish(rec.ctx, "FragmentDropped", time.Now(), metricsctx.UnitCount, float64(1), nil)
		logger.Warn().Str("fragment", r.URL.Path).Msg("fragment queue full, dropping fragment")
	}

	// We write a status ok to ffmpeg so that the transcoder does not crash, which should allow things to eventually catch up.
	w.WriteHeader(http.StatusOK)
}

func (rec *receiver) startServer(ctx context.Context) (err error) {
	httpLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("failed to open listener: %w", err)
	}
	defer func() {
		// close the listener if we exit with an error
		if err != nil {
			_ = httpLn.Close()
		}
	}()

	var lnPort int
	if addr, ok := httpLn.Addr().(*net.TCPAddr); ok {
		lnPort = addr.Port
	} else {
		return fmt.Errorf("failed to determine listener port, got non tcp addr %v", httpLn.Addr())
	}

	rec.endpoint = fmt.Sprintf("http://127.0.0.1:%d/fragment", lnPort)

	mux := goji.NewMux()
	mux.Use(hlog.NewHandler(*log.Ctx(ctx)))
	mux.Use(hlog.URLHandler("url"))
	mux.Use(hlog.AccessHandler(func(r *http.Request, status, size int, duration time.Duration) {
		hlog.FromRequest(r).Debug().Int("status", status).Int("size", size).Dur("duration", duration).Msg("")
	}))
	mux.HandleFunc(pat.Post("/fragment/:fragment"), rec.handleFragment)

	rec.srv = &http.Server{
		BaseContext: func(net.Listener) context.Context { return ctx },
		Handler:     mux,
	}

	go func() {
		defer func() {
			_ = httpLn.Close()
		}()
		rec.srvErr = rec.srv.Serve(httpLn)
	}()

	return nil
}

func (rec *receiver) Err() error {
	return rec.srvErr
}

func (rec *receiver) Shutdown(ctx context.Context) error {
	defer close(rec.fragments)
	err := rec.srv.Shutdown(ctx)
	if err != nil {
		return fmt.Errorf("failed to shut down fragment receiver: %w", err)
	}
	return nil
}
