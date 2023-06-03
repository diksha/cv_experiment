package token

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/golang-jwt/jwt/v5"
	"github.com/stretchr/testify/assert"
)

func generateTestPrivateKey() (string, error) {
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return "", fmt.Errorf("failed to generate private key: %w", err)
	}

	privBytes := x509.MarshalPKCS1PrivateKey(privateKey)
	privPem := pem.EncodeToMemory(&pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: privBytes,
	})

	return string(privPem), nil
}

func TestCreateJWTToken(t *testing.T) {
	testPrivateKey, err := generateTestPrivateKey()
	assert.NoError(t, err)

	appID := int64(12345)
	privateKey, err := loadPrivateKey(testPrivateKey)
	assert.NoError(t, err)

	token, err := createJWTToken(appID, privateKey)
	assert.NoError(t, err)

	parsedToken, err := jwt.Parse(token, func(token *jwt.Token) (interface{}, error) {
		return &privateKey.PublicKey, nil
	})

	assert.NoError(t, err)
	assert.True(t, parsedToken.Valid)

	claims, ok := parsedToken.Claims.(jwt.MapClaims)
	assert.True(t, ok)

	iss, ok := claims["iss"].(float64)
	assert.True(t, ok)
	assert.Equal(t, float64(appID), iss)
}

func TestCreateGitHubToken(t *testing.T) {
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, err := io.ReadAll(r.Body)
		assert.NoError(t, err)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		fmt.Fprintln(w, `{"token": "test-token", "expires_at": "2023-04-04T12:00:00Z"}`)
	}))
	defer testServer.Close()

	testPrivateKey, err := generateTestPrivateKey()
	assert.NoError(t, err)

	// Replace GitHub API URL with test server URL
	oldURL := "https://api.github.com/app/installations/"
	newURL := testServer.URL + "/app/installations/"
	http.DefaultTransport = RewriteTransport{oldURL, newURL, http.DefaultTransport}

	token, err := CreateGitHubToken(12345, "67890", testPrivateKey)
	if err != nil {
		t.Logf("Error: %v", err)
	}
	assert.NoError(t, err)
	assert.Equal(t, "test-token", token)
}

type RewriteTransport struct {
	oldURL     string
	newURL     string
	underlying http.RoundTripper
}

func (t RewriteTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	newURL, err := url.Parse(strings.Replace(req.URL.String(), t.oldURL, t.newURL, 1))
	if err != nil {
		return nil, fmt.Errorf("failed to parse URL in RewriteTransport: %w", err)
	}
	req.URL = newURL
	resp, err := t.underlying.RoundTrip(req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform RoundTrip in RewriteTransport: %w", err)
	}
	return resp, nil
}
