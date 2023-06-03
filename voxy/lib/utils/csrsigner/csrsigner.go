// Package csrsigner contains a certificate signer capable of signing CSRs with the configured CA Key Pair
package csrsigner

import (
	"crypto/rand"
	"crypto/x509"
	"fmt"
	"math/big"
	"time"

	"golang.org/x/crypto/ssh"

	"github.com/voxel-ai/voxel/lib/utils/x509pem"
)

// Signer is capable of Signing CSRs given the CA it has
type Signer struct {
	RootCA *x509.Certificate
	Certs  []*x509.Certificate
	Key    any
}

// LoadFromPEMsWithEncryptedKey attempts load a Signer with the passed in encrypted CA key pair
func LoadFromPEMsWithEncryptedKey(rootPEM, certPEM, keyPEM, password []byte) (*Signer, error) {
	rootCerts, err := x509pem.Decode[[]*x509.Certificate](rootPEM)
	if err != nil {
		return nil, fmt.Errorf("invalid root certificate: %w", err)
	}

	if len(rootCerts) != 1 {
		return nil, fmt.Errorf("got %d root certs, expected 1", len(rootCerts))
	}
	rootCert := rootCerts[0]

	certs, err := x509pem.Decode[[]*x509.Certificate](certPEM)
	if err != nil {
		return nil, fmt.Errorf("invalid certificate: %w", err)
	}

	if len(certs) == 0 {
		return nil, fmt.Errorf("no certificates in certificate pem")
	}

	key, err := ssh.ParseRawPrivateKeyWithPassphrase(keyPEM, password)
	if err != nil {
		return nil, fmt.Errorf("failed to parse private key: %w", err)
	}

	return &Signer{
		RootCA: rootCert,
		Certs:  certs,
		Key:    key,
	}, nil
}

// Sign will sign the passed in CSR ignoring all but the key and Subject.
// The leaf and intermediate signing certificates are returned
func (s *Signer) Sign(req *x509.CertificateRequest) ([]*x509.Certificate, error) {
	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return nil, fmt.Errorf("failed to generate certificate serial number: %w", err)
	}

	template := &x509.Certificate{
		SerialNumber:          serialNumber,
		Subject:               req.Subject,
		NotBefore:             time.Now(),
		NotAfter:              time.Now().AddDate(0, 0, 1),
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	certBytes, err := x509.CreateCertificate(rand.Reader, template, s.Certs[0], req.PublicKey, s.Key)
	if err != nil {
		return nil, fmt.Errorf("failed to generate certificate: %w", err)
	}

	cert, err := x509.ParseCertificate(certBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse generated certificate: %w", err)
	}

	return append([]*x509.Certificate{cert}, s.Certs...), nil
}
