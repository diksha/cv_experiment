// Package token allows to fetch a github token from a github app.
package token

import (
	"crypto/rsa"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

type tokenResponse struct {
	Token     string    `json:"token"`
	ExpiresAt time.Time `json:"expires_at"`
}

// CreateGitHubToken creates a temporary github access token for the given github app.
func CreateGitHubToken(appID int64, installationID, privateKey string) (string, error) {
	parsedKey, err := loadPrivateKey(privateKey)
	if err != nil {
		return "", fmt.Errorf("failed to load private key: %w", err)
	}
	jwtToken, err := createJWTToken(appID, parsedKey)
	if err != nil {
		return "", fmt.Errorf("failed to create JWT token: %w", err)
	}

	client := &http.Client{
		Transport: &jwtTransport{
			transport: http.DefaultTransport,
			jwtToken:  jwtToken,
		},
	}

	resp, err := client.Post("https://api.github.com/app/installations/"+installationID+"/access_tokens", "application/json", nil)
	if err != nil {
		return "", fmt.Errorf("failed to send POST request: %w", err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			log.Printf("Failed to close response body: %v", closeErr)
		}
	}()

	if resp.StatusCode != http.StatusCreated {
		return "", fmt.Errorf("failed to create token: %s", resp.Status)
	}

	var tokenResp tokenResponse
	err = json.NewDecoder(resp.Body).Decode(&tokenResp)
	if err != nil {
		return "", fmt.Errorf("failed to decode token response: %w", err)
	}

	return tokenResp.Token, nil
}

func createJWTToken(appID int64, privateKey *rsa.PrivateKey) (string, error) {
	token := jwt.NewWithClaims(jwt.SigningMethodRS256, jwt.MapClaims{
		"iat": time.Now().Unix(),
		"exp": time.Now().Add(10 * time.Minute).Unix(),
		"iss": appID,
	})
	signedToken, err := token.SignedString(privateKey)
	if err != nil {
		return "", fmt.Errorf("failed to sign JWT token: %w", err)
	}
	return signedToken, nil
}

func loadPrivateKey(privateKey string) (*rsa.PrivateKey, error) {
	parsedKey, err := jwt.ParseRSAPrivateKeyFromPEM([]byte(privateKey))
	if err != nil {
		return nil, fmt.Errorf("failed to parse RSA private key: %w", err)
	}
	return parsedKey, nil
}

type jwtTransport struct {
	transport http.RoundTripper
	jwtToken  string
}

func (t *jwtTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if t.transport == nil {
		return nil, fmt.Errorf("no transport specified: %w", http.ErrSkipAltProtocol)
	}
	req.Header.Set("Authorization", "Bearer "+t.jwtToken)
	req.Header.Set("Accept", "application/vnd.github+json")
	response, err := t.transport.RoundTrip(req)
	if err != nil {
		return nil, fmt.Errorf("round trip error: %w", err)
	}
	return response, nil
}
