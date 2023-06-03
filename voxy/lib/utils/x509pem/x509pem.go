// Package x509pem provides utilities for encoding/decoding PEM files for use
// with the x509 package
package x509pem

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
)

var (
	certPEMBlockType    = "CERTIFICATE"
	certReqPEMBlockType = "CERTIFICATE REQUEST"
	ecKeyPEMBlockType   = "EC PRIVATE KEY"
	rsaKeyPEMBlockType  = "RSA PRIVATE KEY"
)

// ErrNoPEMFound indicates no more PEM data was availble in the input
var ErrNoPEMFound = errors.New("no PEM data found")

// Encodable is the sent of PEM encodable types this library can encode
type Encodable interface {
	~*rsa.PrivateKey |
		~*ecdsa.PrivateKey |
		~[]*x509.Certificate
}

// Decodable is the set of PEM encodable types this library can decode
type Decodable interface {
	Encodable | ~*x509.CertificateRequest
}

// Encode will take the passed in PEM encodable type and return PEM bytes
func Encode[V Encodable](val V) ([]byte, error) {
	switch val := any(val).(type) {
	case *rsa.PrivateKey:
		return encodeRSAKeyPEM(val), nil
	case *ecdsa.PrivateKey:
		return encodeECKeyPEM(val)
	case []*x509.Certificate:
		return encodeCertsPEM(val)
	}

	return nil, fmt.Errorf("unsupported type: %T", val)
}

// Decode will accept the passed in PEM bytes and decode to the specified Encodable type
func Decode[V Decodable](data []byte) (V, error) {
	val, rem, err := decodeOnce[V](data)
	if err != nil {
		return nil, err
	}

	if _, ok := any(V(nil)).([]*x509.Certificate); !ok {
		// for any of the non-list types, just return the first element
		return val, err
	}

	certs := any(val).([]*x509.Certificate)

	for {
		var next []*x509.Certificate
		next, rem, err = decodeOnce[[]*x509.Certificate](rem)
		if errors.Is(err, ErrNoPEMFound) {
			return any(certs).(V), nil
		} else if err != nil {
			return nil, err
		}
		certs = append(certs, next...)
	}
}

func decodeRaw[V Decodable](raw []byte) (val V, err error) {
	switch any(V(nil)).(type) {
	case *rsa.PrivateKey:
		var key *rsa.PrivateKey
		key, err = x509.ParsePKCS1PrivateKey(raw)
		val = any(key).(V)
	case *ecdsa.PrivateKey:
		var key *ecdsa.PrivateKey
		key, err = x509.ParseECPrivateKey(raw)
		val = any(key).(V)
	case []*x509.Certificate:
		var cert *x509.Certificate
		cert, err = x509.ParseCertificate(raw)
		val = any([]*x509.Certificate{cert}).(V)
	case *x509.CertificateRequest:
		var req *x509.CertificateRequest
		req, err = x509.ParseCertificateRequest(raw)
		val = any(req).(V)
	default:
		err = fmt.Errorf("unsupported decode type %T", val)
	}
	return val, err
}

func decodeOnce[V Decodable](data []byte) (V, []byte, error) {
	var expectedType string

	switch any(V(nil)).(type) {
	case *rsa.PrivateKey:
		expectedType = rsaKeyPEMBlockType
	case *ecdsa.PrivateKey:
		expectedType = ecKeyPEMBlockType
	case []*x509.Certificate:
		expectedType = certPEMBlockType
	case *x509.CertificateRequest:
		expectedType = certReqPEMBlockType
	}

	block, rem := pem.Decode(data)
	if block == nil {
		return nil, nil, ErrNoPEMFound
	}

	if block.Type != expectedType {
		return nil, nil, fmt.Errorf("got block type %q, expected %q", block.Type, expectedType)
	}

	val, err := decodeRaw[V](block.Bytes)
	if err != nil {
		return nil, nil, err
	}

	return val, rem, err
}

func encodeECKeyPEM(key *ecdsa.PrivateKey) ([]byte, error) {
	der, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal private key: %w", err)
	}

	return pem.EncodeToMemory(&pem.Block{
		Type:  ecKeyPEMBlockType,
		Bytes: der,
	}), nil
}

func encodeRSAKeyPEM(key *rsa.PrivateKey) []byte {
	return pem.EncodeToMemory(&pem.Block{
		Type:  rsaKeyPEMBlockType,
		Bytes: x509.MarshalPKCS1PrivateKey(key),
	})
}

func encodeCertsPEM(certs []*x509.Certificate) ([]byte, error) {
	var out bytes.Buffer
	for _, cert := range certs {
		err := pem.Encode(&out, &pem.Block{
			Type:  certPEMBlockType,
			Bytes: cert.Raw,
		})
		if err != nil {
			return nil, fmt.Errorf("PEM encoding error: %w", err)
		}
	}
	return out.Bytes(), nil
}
