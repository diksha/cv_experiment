package main

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cristalhq/aconfig"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/lib/utils/mtlstest"
)

func testlogger(t *testing.T) zerolog.Logger {
	return zerolog.New(zerolog.NewConsoleWriter(zerolog.ConsoleTestWriter(t)))
}

func okhandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
}

func defaultConfig(t *testing.T) Config {
	var cfg Config
	err := aconfig.LoaderFor(&cfg, aconfig.Config{
		SkipEnv:   true,
		SkipFlags: true,
		SkipFiles: true,
	}).Load()
	require.NoError(t, err, "must load default config")
	return cfg
}

func mustWriteFile(t *testing.T, filename string, data []byte) {
	require.NoErrorf(t, os.WriteFile(filename, data, 0644), "must write file %v", filename)
}

func TestAppCerts(t *testing.T) {
	// set up the logger and context
	logger := testlogger(t)
	ctx := logger.WithContext(context.Background())
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	logger.Debug().Msg("starting app test")

	logger.Debug().Msg("starting remote servers")
	// set up two remote tls servers
	remotea := httptest.NewTLSServer(okhandler())
	defer remotea.Close()

	remoteb := httptest.NewTLSServer(okhandler())
	defer remoteb.Close()

	// get the hostname for remotea so we can add it to the unauthed allowlist
	remoteaURL, err := url.Parse(remotea.URL)
	require.NoError(t, err, "must parse remote server url")

	// Generate some certificates
	logger.Debug().Msg("generating root ca cert")
	tmpdir, err := os.MkdirTemp("", "pinholeserver-test")
	require.NoError(t, err, "must make pinhole server cert directory")
	defer func() { _ = os.RemoveAll(tmpdir) }()

	rootPath := filepath.Join(tmpdir, "root.crt")
	certPath := filepath.Join(tmpdir, "cert.crt")
	keyPath := filepath.Join(tmpdir, "key.pem")

	certs := mtlstest.MustGen()
	mustWriteFile(t, rootPath, certs.RootPEM)
	mustWriteFile(t, certPath, certs.ServerPEM)
	mustWriteFile(t, keyPath, certs.ServerKeyPEM)

	// configure the main app
	cfg := defaultConfig(t)
	cfg.Addr = "127.0.0.1:0"
	cfg.TLS.RootCA = rootPath
	cfg.TLS.Cert = certPath
	cfg.TLS.Key = keyPath

	app := &App{
		AllowUnauthenticatedAddrs: []string{
			remoteaURL.Host,
		},
		Config: cfg,
	}

	err = app.Listen(ctx)
	require.NoError(t, err, "app should listen successfully")
	proxyAddr := fmt.Sprintf("127.0.0.1:%d", app.ln.Addr().(*net.TCPAddr).Port)

	serveDone := make(chan error, 1)
	go func() {
		serveDone <- app.Serve(ctx)
	}()

	// start a simulated client
	// clientCert, err := tls.LoadX509KeyPair(certPath, keyPath)
	// require.NoError(t, err, "must load client certificates")

	proxyTLSConfig := &tls.Config{
		MinVersion: tls.VersionTLS12,
		RootCAs:    certs.CertPool,
	}

	authProxyTLSConfig := &tls.Config{
		MinVersion:   tls.VersionTLS12,
		RootCAs:      certs.CertPool,
		Certificates: []tls.Certificate{certs.Client},
	}

	proxyFn := func(*http.Request) (*url.URL, error) {
		// trunk-ignore(golangci-lint/wrapcheck)
		return url.Parse("https://" + proxyAddr)
	}

	proxyDialFn := func(ctx context.Context, network, addr string) (net.Conn, error) {
		// trunk-ignore(golangci-lint/wrapcheck)
		return tls.Dial(network, addr, proxyTLSConfig)
	}

	authProxyDialFn := func(ctx context.Context, network, addr string) (net.Conn, error) {
		// trunk-ignore(golangci-lint/wrapcheck)
		return tls.Dial(network, addr, authProxyTLSConfig)
	}

	// clientAddr, err := simulateClient(ctx, t, fmt.Sprintf("localhost:%d", app.ln.Addr().(*net.TCPAddr).Port), clientTLSConfig)
	// require.NoError(t, err, "simulate client must start correctly")

	// first, make an https request to the remote using a client that does not preset a tls certificate to a remote
	// that is on the unauthenticated allowlist
	client := remotea.Client()
	client.Transport.(*http.Transport).Proxy = proxyFn
	client.Transport.(*http.Transport).DialTLSContext = proxyDialFn
	client.Transport.(*http.Transport).DisableKeepAlives = true

	logger.Debug().Msg("making unauthenticated request that should succeed")
	resp, err := client.Get(remotea.URL)
	require.NoError(t, err, "unauthenticated request to remote a must not error")
	assert.Equal(t, http.StatusOK, resp.StatusCode, "remotea request status must be 200 ok")

	// now make a request to a remote that is not on the allowlist, this should error
	client = remoteb.Client()
	client.Transport.(*http.Transport).Proxy = proxyFn
	client.Transport.(*http.Transport).DialTLSContext = proxyDialFn
	client.Transport.(*http.Transport).DisableKeepAlives = true

	logger.Debug().Msg("making unauthenticated request that should fail")
	_, err = client.Get(remoteb.URL)
	require.Error(t, err, "unauthenticated request to remote b must error")

	// now make a request to the remote that is not on the allowlist, but with a client that has a valid certificate
	client = remoteb.Client()
	client.Transport.(*http.Transport).Proxy = proxyFn
	client.Transport.(*http.Transport).DialTLSContext = authProxyDialFn
	client.Transport.(*http.Transport).DisableKeepAlives = true

	logger.Debug().Msg("making authenticated request that should succeed")
	resp, err = client.Get(remoteb.URL)
	require.NoError(t, err, "authenticated request to remote b must not error")
	assert.Equal(t, http.StatusOK, resp.StatusCode, "remoteb request status must be 200 ok")

	shutdownCtx, shutdownCancel := context.WithTimeout(ctx, 5*time.Second)
	defer shutdownCancel()

	logger.Debug().Msg("shutting down app")
	require.NoError(t, app.Shutdown(shutdownCtx), "app shoud shut down cleanly")
}
