// Package autocertfetcher provides a tool that will automatically renew certificates provided by autocert
package autocertfetcher

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
)

var (
	autocertFile = "/var/run/autocert.step.sm/site.crt"
	autocertKey  = "/var/run/autocert.step.sm/site.key"
	autocertRoot = "/var/run/autocert.step.sm/root.crt"
)

const (
	tickFrequency = 1 * time.Second
)

// Fetcher pulls certificates from the specified paths, defaulting to paths from smallstep/autocert
type Fetcher struct {
	// RootPath is the path to the Root CA certificate
	// defaults to "/var/run/autocert.step.sm/root.crt"
	RootPath string
	// CertPath is the path to the certificate file
	// defaults to "/var/run/autocert.step.sm/site.crt"
	CertPath string
	// KeyPath is the path to the certificate key file
	// defaults to "/var/run/autocert.step.sm/site.key"
	KeyPath string

	sync.RWMutex
	cert *tls.Certificate
	pool *x509.CertPool
}

func (f *Fetcher) setDefaults() {
	if f.RootPath == "" {
		f.RootPath = autocertRoot
	}

	if f.CertPath == "" {
		f.CertPath = autocertFile
	}

	if f.KeyPath == "" {
		f.KeyPath = autocertKey
	}
}

// Init attempts to load the configured certificate paths
func (f *Fetcher) Init() error {
	f.setDefaults()

	if err := f.loadRootPool(); err != nil {
		return fmt.Errorf("failed to load root cert: %w", err)
	}

	if err := f.loadCertificates(); err != nil {
		return fmt.Errorf("failed to load cert: %w", err)
	}

	return nil
}

// GetRootCAs returns the Root CA pool
func (f *Fetcher) GetRootCAs() *x509.CertPool {
	f.RLock()
	defer f.RUnlock()
	return f.pool
}

// GetCertificate can be passed to a *tls.Config
func (f *Fetcher) GetCertificate(_ *tls.ClientHelloInfo) (*tls.Certificate, error) {
	f.RLock()
	defer f.RUnlock()
	return f.cert, nil
}

func (f *Fetcher) loadRootPool() error {
	f.Lock()
	defer f.Unlock()

	rootPEM, err := os.ReadFile(f.RootPath)
	if err != nil {
		return fmt.Errorf("failed to load root certificate: %w", err)
	}

	pool := x509.NewCertPool()
	if ok := pool.AppendCertsFromPEM(rootPEM); !ok {
		return fmt.Errorf("missing or invalid root certificate")
	}

	f.pool = pool

	return nil
}

func (f *Fetcher) loadCertificates() error {
	f.Lock()
	defer f.Unlock()

	c, err := tls.LoadX509KeyPair(f.CertPath, f.KeyPath)
	if err != nil {
		return fmt.Errorf("failed to load certifficate: %w", err)
	}

	f.cert = &c

	return nil
}

// Run watches for certificate file updates and refreshes the loaded cert
func (f *Fetcher) Run(ctx context.Context) error {
	if f.cert == nil {
		return fmt.Errorf("attempted to run uninitialized cert fetcher")
	}

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("failed to start fs watcher: %w", err)
	}
	defer func() { _ = watcher.Close() }()

	if err := watcher.Add(f.CertPath); err != nil {
		return fmt.Errorf("failed to watch %q: %w", f.CertPath, err)
	}

	if err := watcher.Add(f.KeyPath); err != nil {
		return fmt.Errorf("failed to watch %q: %w", f.KeyPath, err)
	}

	// use a ticker to make sure we don't reload too fast
	shouldReload := true
	reloadTicker := time.NewTicker(tickFrequency)
	defer reloadTicker.Stop()

	for {
		select {
		case _, ok := <-watcher.Events:
			if !ok {
				return fmt.Errorf("watcher ended early")
			}

			// pretty much any event means that we need to reload
			shouldReload = true
		case err, ok := <-watcher.Errors:
			if !ok {
				return fmt.Errorf("cert watcher ended")
			}
			return fmt.Errorf("cert watcher error: %w", err)
		case <-reloadTicker.C:
			if shouldReload {
				if err := f.loadCertificates(); err != nil {
					return fmt.Errorf("failed to reload certificates: %w", err)
				}
				shouldReload = false
			}
		case <-ctx.Done():
			return nil
		}
	}
}
