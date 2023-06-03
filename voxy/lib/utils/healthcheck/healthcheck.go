// Package healthcheck provides a simple http health endpoint for use with kubernetes deployments
package healthcheck

import (
	"context"
	"net"
	"net/http"
	"time"

	"github.com/rs/zerolog/hlog"
	"github.com/rs/zerolog/log"
	"goji.io"
	"goji.io/pat"
)

var defaultAddr = ":8081"

// ListenAndServe responds to healthchecks for kubernetes to know this server is running
//
// This call is intended to just be a fire and forget via `go healthcheck.ListenAndServe(ctx, addr)`
// If addr is empty the default addr of `":8081"` will be used`. To shut this server down, cancel its context.
func ListenAndServe(ctx context.Context, addr string) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	// we always cancel this context in case this function exits unexpectedly
	// so that resources are still cleaned up properly

	if addr == "" {
		addr = defaultAddr
	}

	mux := goji.NewMux()
	mux.HandleFunc(pat.Get("/health"), func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	handler := hlog.AccessHandler(func(r *http.Request, status, size int, duration time.Duration) {
		hlog.FromRequest(r).Info().
			Str("method", r.Method).
			Stringer("url", r.URL).
			Int("status", status).
			Int("size", size).
			Dur("duration", duration).
			Str("protocol", "http").
			Msg("")
	})(mux)

	srv := &http.Server{
		Addr: addr,
		BaseContext: func(net.Listener) context.Context {
			return ctx
		},
		Handler: handler,
	}

	go func() {
		// wait for the base context to be canceled
		<-ctx.Done()

		// create a shutdown context for the shutdown handler
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer shutdownCancel()

		// attempt to shut down, log an error if we fail
		if err := srv.Shutdown(shutdownCtx); err != nil {
			log.Ctx(ctx).Error().Err(err).Msg("failed to shut down healtcheck server")
		}
	}()

	err := srv.ListenAndServe()

	log.Ctx(ctx).Err(err).Msg("heathcheck server exited, app will no longer appear healthy")
}
