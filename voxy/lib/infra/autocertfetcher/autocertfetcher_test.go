package autocertfetcher_test

import (
	"context"
	"crypto/tls"
	"os"
	"testing"
	"time"

	"github.com/madflojo/testcerts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/lib/infra/autocertfetcher"
)

func TestFetcher(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	dir, err := os.MkdirTemp("", "autocertfetcher-test")
	require.NoError(t, err, "must make temp dir")
	defer func() { _ = os.RemoveAll(dir) }()

	firstFile, firstKey, err := testcerts.GenerateCertsToTempFile(dir)
	require.NoError(t, err, "must generate test certs")

	firstCert, err := tls.LoadX509KeyPair(firstFile, firstKey)
	require.NoError(t, err, "must load first cert pair")

	fetcher := &autocertfetcher.Fetcher{
		RootPath: firstFile,
		CertPath: firstFile,
		KeyPath:  firstKey,
	}
	err = fetcher.Init()
	require.NoError(t, err, "must create fetcher")

	currentCert, err := fetcher.GetCertificate(nil)
	require.NoError(t, err, "must get certificate")

	assert.Equal(t, firstCert.Certificate, currentCert.Certificate, "first certificate must load correctly")

	errch := make(chan error, 1)
	go func() {
		errch <- fetcher.Run(ctx)
	}()

	secondFile, secondKey, err := testcerts.GenerateCertsToTempFile(dir)
	require.NoError(t, err, "must generate second cert")
	require.NoError(t, os.Rename(secondFile, firstFile), "must move cert")
	require.NoError(t, os.Rename(secondKey, firstKey), "must move key")

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	lastCert := currentCert
	for lastCert == currentCert {
		select {
		case err := <-errch:
			require.NoError(t, err, "run must not error")
		case <-ticker.C:
			currentCert, err = fetcher.GetCertificate(nil)
			require.NoError(t, err, "must get certificate")
		case <-ctx.Done():
			t.Fatalf("timed out waiting for new certificate")
		}
	}
}
