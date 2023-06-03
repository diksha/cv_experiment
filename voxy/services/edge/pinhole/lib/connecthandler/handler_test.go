package connecthandler_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"testing"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/hlog"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/services/edge/pinhole/lib/connecthandler"
)

func testlogger(t *testing.T) zerolog.Logger {
	return zerolog.New(zerolog.NewConsoleWriter(zerolog.ConsoleTestWriter(t)))
}

func logrequests(logger zerolog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		next = hlog.AccessHandler(func(r *http.Request, status, size int, duration time.Duration) {
			hlog.FromRequest(r).Debug().Int("status", status).Int("size", size).Dur("duration", duration).Msg("request")
		})(next)
		next = hlog.URLHandler("url")(next)
		next = hlog.MethodHandler("method")(next)
		next = hlog.NewHandler(logger)(next)
		return next
	}
}

func TestConnectProxy(t *testing.T) {
	logger := testlogger(t)

	srv := httptest.NewTLSServer(logrequests(logger)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte("{}"))
		require.NoError(t, err, "write must not error")
	})))
	defer srv.Close()

	logger.Debug().Msgf("server url %v", srv.URL)

	resp, err := srv.Client().Get(srv.URL)
	require.NoError(t, err, "unproxied get request should succeed")
	require.Equal(t, http.StatusOK, resp.StatusCode, "unproxied get request should have 200 status")

	var proxyWg sync.WaitGroup
	proxyHandler := logrequests(logger)(connecthandler.New())
	proxySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxyWg.Add(1)
		defer proxyWg.Done()
		proxyHandler.ServeHTTP(w, r)
	}))
	defer proxySrv.Close()
	defer proxyWg.Wait()

	logger.Debug().Msgf("proxy url %v", proxySrv.URL)

	client := srv.Client()
	client.Transport.(*http.Transport).Proxy = func(*http.Request) (*url.URL, error) {
		// trunk-ignore(golangci-lint/wrapcheck): not important in a test
		return url.Parse(proxySrv.URL)
	}
	client.Transport.(*http.Transport).DisableKeepAlives = true

	logger.Debug().Msg("starting get request")
	resp, err = client.Get(srv.URL)
	require.NoError(t, err, "proxy request must succeed")
	defer func() { _ = resp.Body.Close() }()

	logger.Debug().Msg("checking response status")
	// response must be 200 ok
	require.Equal(t, http.StatusOK, resp.StatusCode, "response must be 200 ok")

	logger.Debug().Msg("reading response body")
	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err, "must read request body")
	require.Equal(t, []byte("{}"), body, "body value must be correct")

	logger.Debug().Msg("closing response body")
	require.NoError(t, resp.Body.Close(), "must close body successfully")
}
