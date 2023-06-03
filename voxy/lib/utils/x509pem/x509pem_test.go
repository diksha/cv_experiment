package x509pem_test

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"testing"

	"github.com/madflojo/testcerts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/lib/utils/x509pem"
)

func TestRSAKey(t *testing.T) {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err, "must generate test key")

	data, err := x509pem.Encode(key)
	require.NoError(t, err, "must encode test key")

	decoded, err := x509pem.Decode[*rsa.PrivateKey](data)
	require.NoError(t, err, "decode must succeed")

	assert.True(t, key.Equal(decoded), "keys must match")
}

func TestECKey(t *testing.T) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err, "must generate test key")

	data, err := x509pem.Encode(key)
	require.NoError(t, err, "must encode test key")

	decoded, err := x509pem.Decode[*ecdsa.PrivateKey](data)
	require.NoError(t, err, "must decode key")

	assert.True(t, key.Equal(decoded), "keys must match")
}

func TestCertificate(t *testing.T) {
	rawCert, _, err := testcerts.GenerateCerts()
	require.NoError(t, err, "must generate test cert")

	cert1, err := x509pem.Decode[[]*x509.Certificate](rawCert)
	require.NoError(t, err, "must parse test cert")

	rawCert, _, err = testcerts.GenerateCerts()
	require.NoError(t, err, "must generate test cert")

	cert2, err := x509pem.Decode[[]*x509.Certificate](rawCert)
	require.NoError(t, err, "must generate test cert")

	certs := append(cert1, cert2...)

	data, err := x509pem.Encode(certs)
	require.NoError(t, err, "must encode test certs")

	decoded, err := x509pem.Decode[[]*x509.Certificate](data)
	require.NoError(t, err, "must decode test certs")

	assert.Len(t, decoded, 2, "must find 2 certs")
	assert.True(t, certs[0].Equal(decoded[0]), "first cert matches")
	assert.True(t, certs[1].Equal(decoded[1]), "second cert matches")
}

func TestCertificateRequest(t *testing.T) {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err, "must generate test key")

	req := &x509.CertificateRequest{
		Subject: pkix.Name{CommonName: "some-test"},
	}

	rawCsr, err := x509.CreateCertificateRequest(rand.Reader, req, key)
	require.NoError(t, err)

	csr := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE REQUEST",
		Bytes: rawCsr,
	})

	decoded, err := x509pem.Decode[*x509.CertificateRequest](csr)
	require.NoError(t, err, "must decode certificate request")
	assert.Equal(t, req.Subject.CommonName, decoded.Subject.CommonName, "common name must match")
}
