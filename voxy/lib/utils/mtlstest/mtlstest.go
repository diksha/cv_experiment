// Package mtlstest provides a simple utility for generating test certificates to
// allow for unit testing of mTLS enabled https servers
package mtlstest

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"time"
)

// Certificates is a set of certificates pre-configured for mTLS
type Certificates struct {
	CertPool *x509.CertPool
	Root     tls.Certificate
	Client   tls.Certificate
	Server   tls.Certificate

	RootPEM    []byte
	RootKeyPEM []byte

	ClientPEM    []byte
	ClientKeyPEM []byte

	ServerPEM    []byte
	ServerKeyPEM []byte
}

// MustGen returns generated TLS certificates for mTLS auth set to only allow localhost
// 127.0.0.1 IP addresses to communicate. These certificates are for testing only
func MustGen() *Certificates {
	template := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "root"},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().AddDate(1, 0, 0),
		IsCA:                  true,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		MaxPathLen:            2,
	}
	caCert, caKey, err := createCertificate(template, template, nil)
	if err != nil {
		panic(err)
	}

	template = &x509.Certificate{
		SerialNumber:          big.NewInt(2),
		Subject:               pkix.Name{CommonName: "server"},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().AddDate(1, 0, 0),
		IsCA:                  true,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IPAddresses:           []net.IP{net.IPv4(127, 0, 0, 1)},
	}
	srvCert, srvKey, err := createCertificate(template, caCert, caKey)
	if err != nil {
		panic(err)
	}

	template = &x509.Certificate{
		SerialNumber:          big.NewInt(2),
		Subject:               pkix.Name{CommonName: "client"},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().AddDate(1, 0, 0),
		IsCA:                  true,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IPAddresses:           []net.IP{net.IPv4(127, 0, 0, 1)},
	}
	clientCert, clientKey, err := createCertificate(template, caCert, caKey)
	if err != nil {
		panic(err)
	}

	certs := &Certificates{
		RootPEM:      mustEncodeCertPEM([]*x509.Certificate{caCert}),
		RootKeyPEM:   mustEncodeKeyPEM(caKey),
		ServerPEM:    mustEncodeCertPEM([]*x509.Certificate{srvCert}),
		ServerKeyPEM: mustEncodeKeyPEM(srvKey),
		ClientPEM:    mustEncodeCertPEM([]*x509.Certificate{clientCert}),
		ClientKeyPEM: mustEncodeKeyPEM(clientKey),
	}

	rootTLSCertificate, err := tls.X509KeyPair(certs.RootPEM, certs.RootKeyPEM)
	if err != nil {
		panic(err)
	}

	serverTLSCertificate, err := tls.X509KeyPair(certs.ServerPEM, certs.ServerKeyPEM)
	if err != nil {
		panic(err)
	}

	clientTLSCertificate, err := tls.X509KeyPair(certs.ClientPEM, certs.ClientKeyPEM)
	if err != nil {
		panic(err)
	}

	certs.Root = rootTLSCertificate
	certs.Server = serverTLSCertificate
	certs.Client = clientTLSCertificate
	certs.CertPool = x509.NewCertPool()
	certs.CertPool.AddCert(caCert)

	return certs
}

func createCertificate(template, parent *x509.Certificate, priv any) (cert *x509.Certificate, key *rsa.PrivateKey, err error) {
	privKey, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to genrate key: %w", err)
	}

	if template == parent {
		priv = privKey
	}

	certBytes, err := x509.CreateCertificate(rand.Reader, template, parent, &privKey.PublicKey, priv)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create certificate: %w", err)
	}

	cert, err = x509.ParseCertificate(certBytes)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse certificate: %w", err)
	}

	return cert, privKey, nil
}

func mustEncodeCertPEM(certs []*x509.Certificate) []byte {
	var out bytes.Buffer
	for _, cert := range certs {
		err := pem.Encode(&out, &pem.Block{
			Type:  "CERTIFICATE",
			Bytes: cert.Raw,
		})
		if err != nil {
			panic(err)
		}
	}
	return out.Bytes()
}

func mustEncodeKeyPEM(key *rsa.PrivateKey) []byte {
	var out bytes.Buffer
	err := pem.Encode(&out, &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(key),
	})
	if err != nil {
		panic(err)
	}
	return out.Bytes()
}
