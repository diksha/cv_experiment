package mtlstest_test

import (
	"bytes"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/voxel-ai/voxel/lib/utils/mtlstest"
)

var certs *mtlstest.Certificates

func init() {
	certs = mtlstest.MustGen()
}

func serve(t *testing.T, ln net.Listener, testdata []byte) {
	for {
		conn, err := ln.Accept()
		if err != nil {
			t.Logf("listener exited with: %v", err)
			return
		}

		go func() {
			_, _ = io.Copy(io.Discard, conn)
		}()

		_, _ = conn.Write(testdata)
		_ = conn.(*tls.Conn).CloseWrite()
	}
}

func testconn(addr string, clientTLS *tls.Config, testdata []byte) error {
	conn, err := tls.Dial("tcp4", addr, clientTLS)
	if err != nil {
		return fmt.Errorf("failed to dial addr: %w", err)
	}
	defer func() { _ = conn.Close() }()

	readErr := make(chan error, 1)
	writeErr := make(chan error, 1)

	go func() {
		defer close(readErr)
		data, err := io.ReadAll(conn)
		if err != nil {
			readErr <- err
			return
		}

		if !bytes.Equal(data, testdata) {
			readErr <- fmt.Errorf("data read does not match testdata")
		}
	}()

	go func() {
		defer close(writeErr)
		_, err := conn.Write(testdata)
		_ = conn.CloseWrite()
		writeErr <- err
	}()

	if err := <-readErr; err != nil {
		return err
	}

	if err := <-writeErr; err != nil {
		return err
	}

	return nil
}

func TestNoClientCert(t *testing.T) {
	testdata := []byte("a")
	serverTLS := &tls.Config{
		MinVersion:   tls.VersionTLS12,
		Certificates: []tls.Certificate{certs.Server},
	}
	clientTLS := &tls.Config{
		MinVersion: tls.VersionTLS12,
		RootCAs:    certs.CertPool,
	}

	ln, err := tls.Listen("tcp4", "127.0.0.1:0", serverTLS)
	require.NoError(t, err, "must listen")
	defer func() { _ = ln.Close() }()

	go serve(t, ln, testdata)

	err = testconn(ln.Addr().String(), clientTLS, testdata)
	require.NoError(t, err, "connection test must pass")
}

func TestClientCert(t *testing.T) {
	testdata := []byte("a")
	serverTLS := &tls.Config{
		MinVersion:   tls.VersionTLS12,
		Certificates: []tls.Certificate{certs.Server},
		ClientCAs:    certs.CertPool,
		ClientAuth:   tls.RequireAndVerifyClientCert,
	}
	clientTLS := &tls.Config{
		MinVersion:   tls.VersionTLS12,
		RootCAs:      certs.CertPool,
		Certificates: []tls.Certificate{certs.Client},
	}

	ln, err := tls.Listen("tcp4", "127.0.0.1:0", serverTLS)
	require.NoError(t, err, "must listen")
	defer func() { _ = ln.Close() }()

	go serve(t, ln, testdata)

	err = testconn(ln.Addr().String(), clientTLS, testdata)
	require.NoError(t, err, "connection test must pass")
}

func TestInvalidClientCert(t *testing.T) {
	testdata := []byte("a")
	serverTLS := &tls.Config{
		MinVersion:   tls.VersionTLS12,
		Certificates: []tls.Certificate{certs.Server},
		ClientCAs:    certs.CertPool,
		ClientAuth:   tls.RequireAndVerifyClientCert,
	}

	invalidCerts := mtlstest.MustGen()

	// run a test with an invalid cert
	clientTLS := &tls.Config{
		MinVersion:   tls.VersionTLS12,
		RootCAs:      certs.CertPool,
		Certificates: []tls.Certificate{invalidCerts.Client},
	}

	ln, err := tls.Listen("tcp4", "127.0.0.1:0", serverTLS)
	require.NoError(t, err, "must listen")
	defer func() { _ = ln.Close() }()
	addr := ln.Addr().String()

	go serve(t, ln, testdata)

	t.Log("running first connection test")
	err = testconn(addr, clientTLS, testdata)
	require.Error(t, err, "connection test must fail")

	// run a test with no cert at all
	clientTLS = &tls.Config{
		MinVersion: tls.VersionTLS12,
		RootCAs:    certs.CertPool,
	}

	t.Log("running second connection test")
	err = testconn(addr, clientTLS, testdata)
	require.Error(t, err, "connection test must fail")
}
