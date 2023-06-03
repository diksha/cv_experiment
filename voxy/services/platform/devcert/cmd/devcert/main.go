package main

import (
	"context"
	"crypto/x509"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/voxel-ai/voxel/lib/utils/x509pem"
	"github.com/voxel-ai/voxel/services/platform/devcert"
)

var defaultCredsPath = filepath.Join(os.Getenv("HOME"), ".voxel/devcert")
var credsPath = flag.String("creds-path", defaultCredsPath, "path to use for the credentials cache, defaults to $HOME/.voxel/devcert")

const (
	certFile = "client.crt"
	keyFile  = "client.key"
	caFile   = "root.crt"
)

// IsExpired returns true if the previously loaded certificate is expired or doesn't exist
func IsExpired() bool {
	certs, err := loadCerts(filepath.Join(*credsPath, certFile))
	if err != nil {
		return true
	}

	return len(certs) == 0 || time.Now().Add(5*time.Minute).After(certs[0].NotAfter)
}

// loads certificate from the passed in cert file
func loadCerts(certPath string) ([]*x509.Certificate, error) {
	data, err := os.ReadFile(certPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read cert file: %w", err)
	}

	certs, err := x509pem.Decode[[]*x509.Certificate](data)
	if err != nil {
		return nil, fmt.Errorf("failed to load cert: %w", err)
	}

	return certs, nil
}

func main() {
	log.SetFlags(0)
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if !IsExpired() {
		return
	}

	certs, err := devcert.Fetch(ctx)
	if err != nil {
		log.Fatalf("failed to get certs: %v", err)
	}

	if err := os.MkdirAll(*credsPath, 0700); err != nil {
		log.Fatalf("failed to make creds cache dir: %v", err)
	}

	files := map[string][]byte{
		filepath.Join(*credsPath, caFile):   certs.RootCA,
		filepath.Join(*credsPath, certFile): certs.Cert,
		filepath.Join(*credsPath, keyFile):  certs.Key,
	}

	for filename, data := range files {
		log.Printf("Writing %v", filename)
		err := os.WriteFile(filename, data, 0600)
		if err != nil {
			log.Fatalf("failed to write %v: %v", filename, err)
		}
	}
}
