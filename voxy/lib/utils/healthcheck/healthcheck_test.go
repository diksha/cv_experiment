package healthcheck_test

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/lib/utils/healthcheck"
)

func logCtx(ctx context.Context, t *testing.T) context.Context {
	return zerolog.New(zerolog.NewTestWriter(t)).WithContext(ctx)
}

func TestHealthcheck(t *testing.T) {
	ctx := logCtx(context.Background(), t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	timeout := time.After(5 * time.Second)

	done := make(chan struct{})
	go func() {
		defer close(done)
		healthcheck.ListenAndServe(ctx, "127.0.0.1:18181")
	}()

	for {
		select {
		case <-timeout:
			t.Fatal("timed out waiting for health endpoint to become available")
		default:
		}

		resp, err := http.Get("http://127.0.0.1:18181/health")
		if err != nil {
			// retry every 50ms
			time.Sleep(50 * time.Millisecond)
			continue
		}

		require.Equal(t, http.StatusOK, resp.StatusCode, "http status must be 200 OK")
		break
	}

	cancel()

	select {
	case <-timeout:
		t.Fatal("timed out waiting for ListenAndServe to shut down")
	case <-done:
	}
}
