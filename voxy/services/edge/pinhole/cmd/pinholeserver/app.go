package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/hlog"
	"github.com/rs/zerolog/log"
	"golang.org/x/exp/slices"

	"github.com/voxel-ai/voxel/lib/infra/autocertfetcher"
	"github.com/voxel-ai/voxel/lib/utils/healthcheck"
	"github.com/voxel-ai/voxel/services/edge/pinhole/lib/connecthandler"
)

// App is the pinholeserver app
type App struct {
	AllowUnauthenticatedAddrs []string
	Config                    Config

	ln  net.Listener
	srv http.Server

	stopHealtcheck func()

	wg sync.WaitGroup
}

// Listen starts the pinholeserver listener but does not serve any connections yet. Do this before
// signaling that the app is healthy
func (a *App) Listen(ctx context.Context) error {
	logger := log.Ctx(ctx)
	logger.Info().Interface("config", &a.Config).Msg("starting listener")

	ln, err := net.Listen("tcp4", a.Config.Addr)
	if err != nil {
		return fmt.Errorf("http listen error: %w", err)
	}

	a.ln = ln

	// start our healtchecker and use the waitgroup to wait for it to close on shutdown
	healthcheckCtx, cancel := context.WithCancel(ctx)
	a.stopHealtcheck = cancel
	a.wg.Add(1)

	go func() {
		defer cancel()
		defer a.wg.Done()

		healthcheck.ListenAndServe(healthcheckCtx, "")
	}()

	return nil
}

// Serve accepts and serves requests on the already listening app, blocking until the listener is closed
func (a *App) Serve(ctx context.Context) error {
	logger := log.Ctx(ctx)
	handler := a.buildHandler(*logger)

	a.srv = http.Server{
		Addr:    a.Config.Addr,
		Handler: handler,
	}

	if a.Config.DevMode {
		logger.Warn().Msg("running in insecure mode without tls")
		if err := a.srv.Serve(a.ln); err != nil {
			return fmt.Errorf("http serve error: %w", err)
		}
		return nil
	}

	tlsConfig, err := a.getTLSConfig(ctx)
	if err != nil {
		return fmt.Errorf("tls config error: %w", err)
	}
	a.srv.TLSConfig = tlsConfig

	logger.Info().Msg("starting tls listener")
	if err := a.srv.ServeTLS(a.ln, "", ""); err != nil {
		return fmt.Errorf("http server error: %w", err)
	}
	return nil
}

// Shutdown attempts to gracefully shut down the app and wait for all outstanding connections to close
func (a *App) Shutdown(ctx context.Context) error {
	a.stopHealtcheck()

	if err := a.srv.Shutdown(ctx); err != nil {
		return fmt.Errorf("http server shutdown error: %w", err)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		a.wg.Wait()
	}()
	select {
	case <-done:
	case <-ctx.Done():
		return fmt.Errorf("http server shutdown error: %w", ctx.Err())
	}

	return nil
}

func (a *App) getTLSConfig(ctx context.Context) (*tls.Config, error) {
	logger := log.Ctx(ctx)

	tlsConfig := &tls.Config{
		ClientAuth:               tls.VerifyClientCertIfGiven,
		MinVersion:               tls.VersionTLS12,
		CurvePreferences:         []tls.CurveID{tls.CurveP521, tls.CurveP256, tls.CurveP384},
		PreferServerCipherSuites: true,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		},
	}

	if a.Config.TLS.Cert == "" {
		logger.Info().Msg("starting certificate autofetcher as no tls cert was provided")
		fetcher := &autocertfetcher.Fetcher{}
		if err := fetcher.Init(); err != nil {
			return nil, fmt.Errorf("failed to initialize cert fetcher: %w", err)
		}

		go func() {
			err := fetcher.Run(ctx)
			logger.Fatal().Err(err).Msg("cert fetcher exited")
		}()

		tlsConfig.GetCertificate = fetcher.GetCertificate
		tlsConfig.ClientCAs = fetcher.GetRootCAs()
	} else {
		logger.Info().Msg("loading certificates from files")
		caCertFile, err := os.ReadFile(a.Config.TLS.RootCA)
		if err != nil {
			return nil, fmt.Errorf("failed to load root ca certificate %q: %w", a.Config.TLS.RootCA, err)
		}
		caCertPool := x509.NewCertPool()
		caCertPool.AppendCertsFromPEM(caCertFile)

		cert, err := tls.LoadX509KeyPair(a.Config.TLS.Cert, a.Config.TLS.Key)
		if err != nil {
			return nil, fmt.Errorf("failed to load certificate keypair from (%q, %q): %w", a.Config.TLS.Cert, a.Config.TLS.Key, err)
		}

		tlsConfig.Certificates = []tls.Certificate{cert}
		tlsConfig.ClientCAs = caCertPool
	}

	return tlsConfig, nil
}

func (a *App) trackRequests() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			a.wg.Add(1)
			defer a.wg.Done()
			next.ServeHTTP(w, r)
		})
	}
}

func (a *App) buildHandler(logger zerolog.Logger) http.Handler {
	// build these backwards
	var handler http.Handler

	handler = connecthandler.New()
	handler = a.trackRequests()(handler)
	handler = authenticateConnectRequest(a.AllowUnauthenticatedAddrs)(handler)
	handler = hlog.AccessHandler(func(r *http.Request, status, size int, duration time.Duration) {
		hlog.FromRequest(r).Info().Int("status", status).Int("size", size).Dur("duration", duration).Msg("")
	})(handler)
	handler = hlog.URLHandler("url")(handler)
	handler = hlog.MethodHandler("method")(handler)
	handler = hlog.NewHandler(logger)(handler)

	return handler
}

func authenticateConnectRequest(allowlist []string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// we do not support any method other than connect
			if r.Method != http.MethodConnect {
				hlog.FromRequest(r).Error().Msg("invalid request method")
				http.NotFound(w, r)
				return
			}

			// this handler is incompatible with non-tls requests
			// and invalid certificates with no name
			if r.TLS == nil {
				hlog.FromRequest(r).Error().Msg("server only compatible with tls connections")
				http.NotFound(w, r)
				return
			}

			if len(r.TLS.VerifiedChains) == 0 {
				// no client certificate was presented
				// deny requests to domains not on the allowlist
				if !slices.Contains(allowlist, r.URL.Host) {
					hlog.FromRequest(r).Warn().Msg("unauthenticated proxy request to invalid domain")
					http.NotFound(w, r)
					return
				}
			}

			next.ServeHTTP(w, r)
		})
	}
}
