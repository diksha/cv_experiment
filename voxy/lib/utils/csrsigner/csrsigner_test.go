package csrsigner_test

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
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/lib/utils/csrsigner"
)

var testCerts struct {
	caCertPEM   []byte
	caKeyPEM    []byte
	signCertPEM []byte
	signKeyPEM  []byte
	signKeyPass []byte
	srvCertPEM  []byte
	srvKeyPEM   []byte
}

func init() {
	mustGenerateCerts()
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

func mustGenerateCerts() {
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
		Subject:               pkix.Name{CommonName: "intermediate"},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().AddDate(1, 0, 0),
		IsCA:                  true,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		MaxPathLen:            1,
	}
	signingCert, signingKey, err := createCertificate(template, caCert, caKey)
	if err != nil {
		panic(err)
	}

	password := []byte("password")
	// trunk-ignore(golangci-lint/staticcheck): need to test this flow since we can't always control whether keys are encrypted
	encryptedSigningKey, err := x509.EncryptPEMBlock(
		rand.Reader,
		"RSA PRIVATE KEY",
		x509.MarshalPKCS1PrivateKey(signingKey),
		password,
		x509.PEMCipherAES256,
	)
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
	srvCert, srvKey, err := createCertificate(template, signingCert, signingKey)
	if err != nil {
		panic(err)
	}

	testCerts.caCertPEM = mustEncodeCertPEM([]*x509.Certificate{caCert})
	testCerts.caKeyPEM = mustEncodeKeyPEM(caKey)
	testCerts.signCertPEM = mustEncodeCertPEM([]*x509.Certificate{signingCert})
	testCerts.signKeyPEM = pem.EncodeToMemory(encryptedSigningKey)
	testCerts.signKeyPass = password
	testCerts.srvCertPEM = mustEncodeCertPEM([]*x509.Certificate{srvCert, signingCert})
	testCerts.srvKeyPEM = mustEncodeKeyPEM(srvKey)
}

func TestSign(t *testing.T) {
	signer, err := csrsigner.LoadFromPEMsWithEncryptedKey(testCerts.caCertPEM, testCerts.signCertPEM, testCerts.signKeyPEM, testCerts.signKeyPass)
	require.NoError(t, err, "must load cert signer")
	require.NotNil(t, signer.RootCA, "must load root ca")

	privateKey, err := rsa.GenerateKey(rand.Reader, 4096)
	require.NoError(t, err, "must generate private key")

	csr, err := x509.CreateCertificateRequest(rand.Reader, &x509.CertificateRequest{
		Subject: pkix.Name{CommonName: "test-server"},
	}, privateKey)
	require.NoError(t, err, "must create csr")

	req, err := x509.ParseCertificateRequest(csr)
	require.NoError(t, err, "must parse csr")

	assert.Equal(t, "test-server", req.Subject.CommonName, "common name must match")

	signed, err := signer.Sign(req)
	require.NoError(t, err, "must sign certificate")

	assert.Equal(t, "test-server", signed[0].Subject.CommonName, "common name must match")
}

func TestSignedCertificate(t *testing.T) {
	caCertPool := x509.NewCertPool()
	require.True(t, caCertPool.AppendCertsFromPEM(testCerts.caCertPEM), "must apped root cert to pool")

	signer, err := csrsigner.LoadFromPEMsWithEncryptedKey(testCerts.caCertPEM, testCerts.signCertPEM, testCerts.signKeyPEM, testCerts.signKeyPass)
	require.NoError(t, err, "must create signer")

	clientKey, err := rsa.GenerateKey(rand.Reader, 4096)
	require.NoError(t, err, "must generate private key")

	clientCsr, err := x509.CreateCertificateRequest(rand.Reader, &x509.CertificateRequest{
		Subject: pkix.Name{CommonName: "test-server"},
	}, clientKey)
	require.NoError(t, err, "must create csr")

	clientReq, err := x509.ParseCertificateRequest(clientCsr)
	require.NoError(t, err, "must parse csr")

	clientCert, err := signer.Sign(clientReq)
	require.NoError(t, err, "must sign certificate")

	testServer := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	serverPair, err := tls.X509KeyPair(testCerts.srvCertPEM, testCerts.srvKeyPEM)
	require.NoError(t, err, "must load server keypair")

	clientPair, err := tls.X509KeyPair(mustEncodeCertPEM(clientCert), mustEncodeKeyPEM(clientKey))
	require.NoError(t, err, "must load client keypair")

	testServer.TLS = &tls.Config{
		MinVersion:   tls.VersionTLS12,
		ClientCAs:    caCertPool,
		ClientAuth:   tls.RequireAndVerifyClientCert,
		Certificates: []tls.Certificate{serverPair},
	}

	testServer.StartTLS()
	defer testServer.Close()

	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				MinVersion:   tls.VersionTLS12,
				RootCAs:      caCertPool,
				Certificates: []tls.Certificate{clientPair},
			},
		},
	}

	_, err = client.Get(testServer.URL)
	require.NoError(t, err, "get request must be succesful")
}
